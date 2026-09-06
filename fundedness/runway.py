"""Time Runway plan builder.

Turns a Household + retirement year into the year-by-year schedules the
Monte Carlo engine needs (pension income, pre-retirement savings, gross
portfolio draws, spending floor), with UK income tax and CGT applied
per person via TaxModel.

All values are in real (today's pounds) terms.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field

import numpy as np

from fundedness.models.assets import AccountType, AssetClass
from fundedness.models.household import Household
from fundedness.models.liabilities import InflationLinkage, Liability
from fundedness.models.market import MarketModel
from fundedness.models.simulation import SimulationConfig
from fundedness.models.tax import TaxModel
from fundedness.simulate import SimulationResult, run_simulation

# Fraction of a SIPP withdrawal that is tax-free (25% pension commencement lump sum)
SIPP_TAX_FREE_FRACTION = 0.25


@dataclass
class DrawMix:
    """How portfolio withdrawals are split by tax treatment.

    Shares are fractions of each £1 withdrawn, derived from the current
    balance-sheet composition (positive-valued assets only). Held fixed
    over the projection — a simplification, since pots deplete at
    different rates.
    """

    sipp_share: float = 0.0  # UK SIPP: 75% taxable as income
    deferred_share: float = 0.0  # Other tax-deferred: 100% taxable as income
    taxable_share: float = 0.0  # GIA/taxable: CGT on the gain portion
    exempt_share: float = 0.0  # ISA/tax-exempt/cash: no tax
    gain_ratio: float = 0.0  # Unrealised gain as fraction of taxable-account value

    @classmethod
    def from_household(cls, household: Household, default_cost_basis_ratio: float = 0.5) -> "DrawMix":
        sipp = deferred = taxable = exempt = 0.0
        taxable_gain = 0.0
        for a in household.balance_sheet.assets:
            if a.value <= 0:
                continue  # negative "assets" are obligations, not a draw source
            match a.account_type:
                case AccountType.SIPP:
                    sipp += a.value
                case AccountType.TAX_DEFERRED:
                    deferred += a.value
                case AccountType.TAXABLE | AccountType.GENERAL:
                    if a.asset_class == AssetClass.CASH:
                        exempt += a.value  # cash has no gain to tax
                    else:
                        taxable += a.value
                        basis = a.cost_basis if a.cost_basis is not None else a.value * default_cost_basis_ratio
                        taxable_gain += max(0.0, a.value - basis)
                case _:
                    exempt += a.value
        total = sipp + deferred + taxable + exempt
        if total <= 0:
            return cls()
        return cls(
            sipp_share=sipp / total,
            deferred_share=deferred / total,
            taxable_share=taxable / total,
            exempt_share=exempt / total,
            gain_ratio=(taxable_gain / taxable) if taxable > 0 else 0.0,
        )


def draw_tax(gross_draw: float, other_income: float, mix: DrawMix, tax_model: TaxModel) -> float:
    """Tax due on one person's gross portfolio withdrawal, stacked on their other income."""
    if gross_draw <= 0:
        return 0.0
    taxable_income = gross_draw * (
        mix.sipp_share * (1 - SIPP_TAX_FREE_FRACTION) + mix.deferred_share
    )
    income_tax = tax_model.model_copy(update={"other_income": other_income}).compute_income_tax(taxable_income)
    gains = gross_draw * mix.taxable_share * mix.gain_ratio
    # CGT rate depends on how much basic-rate band the income (pension + taxable draw) has used
    cgt = tax_model.model_copy(update={"other_income": other_income + taxable_income}).compute_cgt(gains)
    return income_tax + cgt


def solve_gross_draw(
    net_needed: float, other_income: float, mix: DrawMix, tax_model: TaxModel
) -> tuple[float, float]:
    """Find the gross withdrawal that leaves `net_needed` after tax.

    Returns:
        (gross_draw, tax)
    """
    if net_needed <= 0:
        return 0.0, 0.0
    gross = net_needed
    tax = 0.0
    for _ in range(50):
        tax = draw_tax(gross, other_income, mix, tax_model)
        new_gross = net_needed + tax
        if abs(new_gross - gross) < 0.01:
            gross = new_gross
            break
        gross = new_gross
    return gross, tax


def is_ongoing(liability: Liability) -> bool:
    """True for spending that starts now and runs for life (vs a dated one-off)."""
    return liability.start_year == 0 and liability.end_year is None


def liability_in_year(liability: Liability, yr: int, assumed_inflation: float = 0.025) -> float:
    """Real-terms amount of a liability in projection year `yr` (0 = first year).

    Active for start_year <= yr < end_year (end exclusive, matching
    fundedness.liabilities.generate_liability_schedule). A fixed-nominal
    (inflation_linkage NONE) amount is eroded by inflation; everything else
    is treated as holding its real value.
    """
    end = liability.end_year if liability.end_year is not None else float("inf")
    if not (liability.start_year <= yr < end):
        return 0.0
    amount = liability.annual_amount * liability.probability
    if liability.inflation_linkage == InflationLinkage.NONE:
        amount /= (1 + assumed_inflation) ** (yr + 1)
    return amount


@dataclass
class RunwayPlan:
    """Year-by-year schedules for one runway scenario (real terms)."""

    n_years: int
    starting_age: int
    years_until_retired: int
    retirement_year: int | None
    initial_wealth: float
    annual_spending: float  # ongoing (for-life) spending target
    spending_floor: float  # ongoing essential spending

    # Per-year arrays (length n_years)
    spending_by_year: np.ndarray  # total net consumption target incl. dated costs
    floor_net_by_year: np.ndarray  # essential net consumption incl. dated essential costs
    dated_by_year: np.ndarray  # dated / one-off costs only (net)
    pension_income_by_year: np.ndarray  # gross, all members
    pension_sources: dict[str, np.ndarray]  # name -> gross per-year
    savings_by_year: np.ndarray  # pre-retirement contributions (positive)
    net_spending_by_year: np.ndarray  # portfolio draw (+) or contribution (-) passed to the sim
    floor_by_year: np.ndarray  # gross portfolio-draw floor
    tax_by_year: np.ndarray  # total household tax on draws
    gross_up_by_year: np.ndarray  # gross draw / net draw (1.0 when no draw)
    severance: float = 0.0
    draw_mix: DrawMix = field(default_factory=DrawMix)
    dated_liabilities: list[Liability] = field(default_factory=list)

    @property
    def is_working(self) -> bool:
        return self.years_until_retired > 0


def build_runway_plan(
    household: Household,
    retirement_year: int | None,
    tax_model: TaxModel | None = None,
    *,
    current_year: int | None = None,
    annual_spending: float | None = None,
    spending_floor: float | None = None,
    initial_wealth: float | None = None,
    n_years: int | None = None,
    assumed_inflation: float = 0.025,
) -> RunwayPlan:
    """Build the schedules for a runway simulation.

    Joint model: one portfolio, one spending need. While working, salary
    covers ongoing spending; savings (plus any pension already in payment)
    flow into the portfolio, and dated one-off costs (e.g. university fees)
    are drawn from it. After retirement, spending is met from pensions first
    and the portfolio covers the remainder, grossed up for tax.

    `annual_spending` / `spending_floor` override the *ongoing* spending
    (dated costs are unchanged) so sensitivity analysis can vary one input.
    """
    tax_model = tax_model or TaxModel()
    current_year = current_year or datetime.date.today().year
    n_years = n_years or household.planning_horizon
    initial_wealth = household.total_assets if initial_wealth is None else initial_wealth

    starting_age = household.primary_member.age if household.primary_member else 65
    years_until_retired = max(0, retirement_year - current_year) if retirement_year is not None else 0

    # --- Spending: ongoing (for life) vs dated one-offs ---
    ongoing = household.total_spending if annual_spending is None else annual_spending
    ongoing_floor = household.essential_spending if spending_floor is None else spending_floor
    ongoing_floor = min(ongoing_floor, ongoing)

    dated = [l for l in household.liabilities if not is_ongoing(l)]
    dated_by_year = np.zeros(n_years)
    dated_floor_by_year = np.zeros(n_years)
    for l in dated:
        for yr in range(n_years):
            amt = liability_in_year(l, yr, assumed_inflation)
            dated_by_year[yr] += amt
            if l.is_essential:
                dated_floor_by_year[yr] += amt
    spending_by_year = ongoing + dated_by_year
    floor_net_by_year = ongoing_floor + dated_floor_by_year

    # --- Pension income, per member and per source ---
    pension_income_by_year = np.zeros(n_years)
    pension_sources: dict[str, np.ndarray] = {}
    member_pension_by_year: dict[str, np.ndarray] = {}

    for member in household.members:
        member_total = np.zeros(n_years)
        sp_age = member.social_security_age or 67
        db_start_age = member.pension_start_age or 60

        if member.social_security_annual > 0:
            arr = np.zeros(n_years)
            for yr in range(n_years):
                if member.age + yr + 1 >= sp_age:
                    arr[yr] = member.social_security_annual  # triple-locked: flat in real terms
            pension_sources[f"State Pension ({member.name})"] = arr
            member_total += arr

        if member.pension_annual > 0:
            arr = np.zeros(n_years)
            # A fixed (non-linked) pension loses value to inflation from the
            # later of its start age and today — pension_annual is today's amount
            # if it is already in payment.
            deflate_from_age = max(db_start_age, member.age)
            for yr in range(n_years):
                age_in_year = member.age + yr + 1
                if age_in_year >= db_start_age:
                    if member.pension_inflation_linked:
                        arr[yr] = member.pension_annual
                    else:
                        arr[yr] = member.pension_annual / (1 + assumed_inflation) ** (age_in_year - deflate_from_age)
            pension_sources[f"DB Pension ({member.name})"] = arr
            member_total += arr

        member_pension_by_year[member.name] = member_total
        pension_income_by_year += member_total

    # --- Pre-retirement savings ---
    savings_by_year = np.zeros(n_years)
    for s in household.savings_contributions:
        if s.annual_amount > 0:
            for yr in range(min(years_until_retired, n_years)):
                savings_by_year[yr] += s.annual_amount * (1 + s.growth_rate) ** yr
    # Pension income received while still working is saved (salary covers spending)
    for member in household.members:
        pension = member_pension_by_year[member.name]
        for yr in range(min(years_until_retired, n_years)):
            if pension[yr] > 0:
                savings_by_year[yr] += pension[yr] - tax_model.compute_income_tax(pension[yr])

    # --- Portfolio draws, grossed up for tax per person ---
    mix = DrawMix.from_household(household, tax_model.default_cost_basis_ratio)
    members = household.members or []
    n_members = max(1, len(members))

    def household_gross(net_gap: float, net_floor_gap: float, yr: int) -> tuple[float, float, float]:
        """Gross draw, tax and gross floor for a household net gap, split equally between members
        so both personal allowances / basic-rate bands are used."""
        other_incomes = [member_pension_by_year[m.name][yr] for m in members] or [0.0]
        gross = tax = floor_gross = 0.0
        for other in other_incomes:
            g, t = solve_gross_draw(net_gap / n_members, other, mix, tax_model)
            gross += g
            tax += t
            fg, _ = solve_gross_draw(net_floor_gap / n_members, other, mix, tax_model)
            floor_gross += fg
        return gross, tax, floor_gross

    net_spending_by_year = np.zeros(n_years)
    floor_by_year = np.zeros(n_years)
    tax_by_year = np.zeros(n_years)
    gross_up_by_year = np.ones(n_years)

    for yr in range(n_years):
        if yr < years_until_retired:
            # Salary covers ongoing spending; only dated costs come from the portfolio
            gap, floor_gap = dated_by_year[yr], dated_floor_by_year[yr]
            gross, tax, floor_gross = household_gross(gap, floor_gap, yr)
            net_spending_by_year[yr] = gross - savings_by_year[yr]
        else:
            pensions_net = sum(
                member_pension_by_year[m.name][yr] - tax_model.compute_income_tax(member_pension_by_year[m.name][yr])
                for m in members
            )
            gap = max(0.0, spending_by_year[yr] - pensions_net)
            floor_gap = max(0.0, floor_net_by_year[yr] - pensions_net)
            gross, tax, floor_gross = household_gross(gap, floor_gap, yr)
            net_spending_by_year[yr] = gross
        floor_by_year[yr] = floor_gross
        tax_by_year[yr] = tax
        gross_up_by_year[yr] = gross / gap if gap > 0 else 1.0

    # --- Thai severance: one-off inflow in the retirement year ---
    severance = sum(m.thai_severance_gbp(retirement_year) for m in members)
    if severance > 0 and 0 < years_until_retired < n_years:
        net_spending_by_year[years_until_retired] -= severance

    return RunwayPlan(
        n_years=n_years,
        starting_age=starting_age,
        years_until_retired=years_until_retired,
        retirement_year=retirement_year,
        initial_wealth=initial_wealth,
        annual_spending=ongoing,
        spending_floor=ongoing_floor,
        spending_by_year=spending_by_year,
        floor_net_by_year=floor_net_by_year,
        dated_by_year=dated_by_year,
        pension_income_by_year=pension_income_by_year,
        pension_sources=pension_sources,
        savings_by_year=savings_by_year,
        net_spending_by_year=net_spending_by_year,
        floor_by_year=floor_by_year,
        tax_by_year=tax_by_year,
        gross_up_by_year=gross_up_by_year,
        severance=severance,
        draw_mix=mix,
        dated_liabilities=dated,
    )


def run_runway(
    plan: RunwayPlan,
    market_model: MarketModel,
    *,
    n_simulations: int = 5000,
    stock_allocation: float = 0.6,
    return_model: str = "lognormal",
    market_source: str = "uk",
    return_shift: float = 0.0,
    random_seed: int | None = 42,
) -> SimulationResult:
    """Run the Monte Carlo simulation for a plan."""
    config = SimulationConfig(
        n_simulations=n_simulations,
        n_years=plan.n_years,
        market_model=market_model,
        random_seed=random_seed,
        return_model=return_model,
        market_source=market_source,
    )
    return run_simulation(
        initial_wealth=plan.initial_wealth,
        annual_spending=plan.net_spending_by_year,
        config=config,
        stock_weight=stock_allocation,
        spending_floor=plan.floor_by_year,
        return_shift=return_shift,
    )


def total_spending_percentiles(plan: RunwayPlan, result: SimulationResult) -> dict[str, np.ndarray]:
    """Convert gross portfolio-withdrawal percentiles into total household consumption.

    Pre-retirement: salary covers ongoing spending and dated costs are assumed paid.
    Post-retirement: net withdrawal + pensions, dropping to pension-only when the
    portfolio is exhausted.
    """
    out: dict[str, np.ndarray] = {}
    for key, vals in result.spending_percentiles.items():
        total = np.zeros_like(vals)
        for yr in range(len(vals)):
            full = plan.spending_by_year[yr]
            if yr < plan.years_until_retired:
                total[yr] = full
                continue
            pension = plan.pension_income_by_year[yr]
            target = plan.net_spending_by_year[yr]
            actual = vals[yr]
            if target < 0:
                total[yr] = full  # one-off inflow year (severance)
            elif actual <= 0:
                total[yr] = pension
            elif actual >= target * 0.99:
                total[yr] = full
            else:
                total[yr] = actual / plan.gross_up_by_year[yr] + pension
        out[key] = total
    return out
