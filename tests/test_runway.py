"""Tests for the Time Runway plan builder and per-person UK tax gross-up."""

import numpy as np
import pytest

from fundedness.models.assets import AccountType, Asset, AssetClass, BalanceSheet
from fundedness.models.household import Household, Person, SavingsContribution
from fundedness.models.liabilities import Liability, LiabilityType
from fundedness.models.market import MarketModel
from fundedness.models.tax import TaxModel
from fundedness.runway import (
    DrawMix,
    build_runway_plan,
    draw_tax,
    is_ongoing,
    liability_in_year,
    run_runway,
    solve_gross_draw,
    total_spending_percentiles,
)
from fundedness.models.liabilities import InflationLinkage
from fundedness.simulate import generate_returns

CURRENT_YEAR = 2026


def _household(members=None, assets=None, savings=None, spending=(50_000, 20_000)):
    members = members or [
        Person(name="A", age=54, social_security_annual=12_000, social_security_age=67,
               pension_annual=9_500, pension_start_age=60, pension_inflation_linked=False, is_primary=True),
        Person(name="B", age=51, social_security_annual=12_000, social_security_age=67,
               pension_annual=1_000, pension_start_age=65, pension_inflation_linked=True, is_primary=False),
    ]
    assets = assets or [
        Asset(name="SIPP", value=220_000, account_type=AccountType.SIPP, asset_class=AssetClass.STOCKS),
        Asset(name="GIA", value=250_000, account_type=AccountType.GENERAL, asset_class=AssetClass.STOCKS,
              cost_basis=150_000),
        Asset(name="Cash", value=20_000, account_type=AccountType.TAXABLE, asset_class=AssetClass.CASH),
        Asset(name="Fees", value=-120_000, account_type=AccountType.TAXABLE, asset_class=AssetClass.STOCKS),
    ]
    essential, discretionary = spending
    return Household(
        members=members,
        balance_sheet=BalanceSheet(assets=assets),
        liabilities=[
            Liability(name="Essential", liability_type=LiabilityType.ESSENTIAL_SPENDING,
                      annual_amount=essential, is_essential=True),
            Liability(name="Discretionary", liability_type=LiabilityType.DISCRETIONARY_SPENDING,
                      annual_amount=discretionary, is_essential=False),
        ],
        savings_contributions=savings or [],
    )


class TestDrawMix:
    def test_shares_ignore_negative_assets_and_treat_cash_as_exempt(self):
        mix = DrawMix.from_household(_household())
        total = 220_000 + 250_000 + 20_000
        assert mix.sipp_share == pytest.approx(220_000 / total)
        assert mix.taxable_share == pytest.approx(250_000 / total)
        assert mix.exempt_share == pytest.approx(20_000 / total)
        assert mix.deferred_share == 0
        assert mix.gain_ratio == pytest.approx(100_000 / 250_000)

    def test_empty_balance_sheet(self):
        assert DrawMix.from_household(_household(assets=[Asset(name="x", value=0)])) == DrawMix()


class TestTaxGrossUp:
    def test_pure_sipp_draw_inside_personal_allowance_is_tax_free(self):
        mix = DrawMix(sipp_share=1.0)
        # 75% of £16,000 = £12,000 taxable, under the £12,570 allowance
        assert draw_tax(16_000, 0.0, mix, TaxModel()) == 0.0

    def test_pure_sipp_draw_basic_rate(self):
        mix = DrawMix(sipp_share=1.0)
        # £40,000 gross → £30,000 taxable → (30,000 - 12,570) * 20%
        assert draw_tax(40_000, 0.0, mix, TaxModel()) == pytest.approx((30_000 - 12_570) * 0.20)

    def test_other_income_uses_up_allowance(self):
        mix = DrawMix(sipp_share=1.0)
        tm = TaxModel()
        assert draw_tax(20_000, 12_570, mix, tm) == pytest.approx(15_000 * 0.20)

    def test_exempt_draw_has_no_tax(self):
        assert draw_tax(100_000, 50_000, DrawMix(exempt_share=1.0), TaxModel()) == 0.0

    def test_gia_draw_pays_cgt_on_gains_above_exemption(self):
        mix = DrawMix(taxable_share=1.0, gain_ratio=0.4)
        # £30,000 draw → £12,000 gain → (12,000 - 3,000) * 18% (basic-rate band unused)
        assert draw_tax(30_000, 0.0, mix, TaxModel()) == pytest.approx(9_000 * 0.18)

    def test_solve_gross_draw_nets_to_target(self):
        mix = DrawMix.from_household(_household())
        tm = TaxModel()
        for net in (5_000, 25_000, 60_000):
            gross, tax = solve_gross_draw(net, 12_000, mix, tm)
            assert gross - tax == pytest.approx(net, abs=0.05)
            assert tax == pytest.approx(draw_tax(gross, 12_000, mix, tm), abs=0.05)

    def test_zero_or_negative_need_draws_nothing(self):
        assert solve_gross_draw(0, 0, DrawMix(sipp_share=1), TaxModel()) == (0.0, 0.0)
        assert solve_gross_draw(-5, 0, DrawMix(sipp_share=1), TaxModel()) == (0.0, 0.0)


class TestPlanSchedules:
    def test_pre_retirement_years_are_contributions(self):
        hh = _household(savings=[SavingsContribution(name="Salary", annual_amount=50_000, growth_rate=0.03)])
        plan = build_runway_plan(hh, 2033, current_year=CURRENT_YEAR)
        assert plan.years_until_retired == 7
        # Contributions grow at the real growth rate
        assert plan.savings_by_year[0] == pytest.approx(50_000)
        assert plan.savings_by_year[1] == pytest.approx(50_000 * 1.03)
        assert np.all(plan.net_spending_by_year[:7] < 0)
        assert np.all(plan.floor_by_year[:7] == 0)
        # First retirement year is a draw
        assert plan.net_spending_by_year[7] > 0

    def test_pension_already_in_payment_while_working_is_saved(self):
        # A's DB pension starts at 60 (year index 5, age-in-year 60); retirement at 61 (index 6)
        hh = _household()
        plan = build_runway_plan(hh, 2033, current_year=CURRENT_YEAR)
        # Age in year yr is 54 + yr + 1, so age 60 => yr 5, and 9,500 < personal allowance => untaxed
        assert plan.savings_by_year[5] == pytest.approx(9_500)
        assert plan.savings_by_year[4] == 0

    def test_fixed_pension_is_deflated_from_start_age(self):
        plan = build_runway_plan(_household(), 2033, current_year=CURRENT_YEAR)
        db = plan.pension_sources["DB Pension (A)"]
        assert db[4] == 0  # age 59
        assert db[5] == pytest.approx(9_500)  # age 60, first year, undeflated
        assert db[6] == pytest.approx(9_500 / 1.025)
        assert db[15] == pytest.approx(9_500 / 1.025 ** 10)

    def test_fixed_pension_already_in_payment_is_deflated_from_today(self):
        member = Person(name="R", age=65, pension_annual=10_000, pension_start_age=60,
                        pension_inflation_linked=False, social_security_annual=0)
        plan = build_runway_plan(_household(members=[member]), None, current_year=CURRENT_YEAR)
        db = plan.pension_sources["DB Pension (R)"]
        # pension_annual is today's amount: first projected year is one year of inflation
        assert db[0] == pytest.approx(10_000 / 1.025)
        assert db[1] == pytest.approx(10_000 / 1.025 ** 2)

    def test_linked_pension_is_flat_in_real_terms(self):
        plan = build_runway_plan(_household(), 2033, current_year=CURRENT_YEAR)
        db = plan.pension_sources["DB Pension (B)"]
        # B is 51: age 65 is yr index 13
        assert db[12] == 0
        assert np.allclose(db[13:], 1_000)

    def test_state_pension_starts_at_claim_age(self):
        plan = build_runway_plan(_household(), 2033, current_year=CURRENT_YEAR)
        sp = plan.pension_sources["State Pension (A)"]
        assert sp[11] == 0  # age 66
        assert sp[12] == 12_000  # age 67

    def test_draw_covers_gap_after_tax(self):
        hh = _household()
        tm = TaxModel()
        plan = build_runway_plan(hh, 2033, tm, current_year=CURRENT_YEAR)
        yr = plan.n_years - 1  # all pensions in payment
        pensions = plan.pension_income_by_year[yr]
        # Net-of-tax pensions + net-of-tax draw == spending
        pension_tax = sum(
            tm.compute_income_tax(src[yr])
            for name, src in plan.pension_sources.items()
        )
        # per-member pension tax stacks state + DB, so recompute per member
        member_pension_tax = 0.0
        for m in hh.members:
            p = sum(v[yr] for k, v in plan.pension_sources.items() if f"({m.name})" in k)
            member_pension_tax += tm.compute_income_tax(p)
        net_income = pensions - member_pension_tax + plan.net_spending_by_year[yr] - plan.tax_by_year[yr]
        assert net_income == pytest.approx(plan.annual_spending, abs=0.1)
        assert plan.gross_up_by_year[yr] > 1.0

    def test_floor_is_gross_of_tax_and_below_draw(self):
        plan = build_runway_plan(_household(), 2033, current_year=CURRENT_YEAR)
        post = slice(plan.years_until_retired + 1, None)
        assert np.all(plan.floor_by_year[post] <= plan.net_spending_by_year[post] + 1e-6)
        assert np.all(plan.floor_by_year[post] > 0)

    def test_pensions_exceeding_spending_means_no_draw(self):
        member = Person(name="Rich", age=70, social_security_annual=40_000, social_security_age=67,
                        pension_annual=40_000, pension_start_age=60, pension_inflation_linked=True)
        plan = build_runway_plan(_household(members=[member], spending=(30_000, 10_000)), None,
                                 current_year=CURRENT_YEAR)
        assert np.all(plan.net_spending_by_year == 0)
        assert np.all(plan.gross_up_by_year == 1.0)

    def test_severance_is_injected_in_retirement_year(self):
        member = Person(name="A", age=54, thai_severance_enabled=True, thai_monthly_wage_thb=300_000,
                        thai_employment_start_year=2005, thai_thb_per_gbp=44.0)
        hh = _household(members=[member])
        plan = build_runway_plan(hh, 2033, current_year=CURRENT_YEAR)
        assert plan.severance == pytest.approx(member.thai_severance_gbp(2033))
        without = build_runway_plan(
            _household(members=[Person(name="A", age=54)]), 2033, current_year=CURRENT_YEAR
        )
        idx = plan.years_until_retired
        assert plan.net_spending_by_year[idx] == pytest.approx(
            without.net_spending_by_year[idx] - plan.severance
        )

    def test_overrides(self):
        hh = _household()
        base = build_runway_plan(hh, 2033, current_year=CURRENT_YEAR)
        more = build_runway_plan(hh, 2033, current_year=CURRENT_YEAR, annual_spending=base.annual_spending + 10_000)
        assert more.net_spending_by_year[-1] > base.net_spending_by_year[-1]
        longer = build_runway_plan(hh, 2033, current_year=CURRENT_YEAR, n_years=base.n_years + 5)
        assert longer.n_years == base.n_years + 5
        assert len(longer.net_spending_by_year) == longer.n_years


class TestRunRunway:
    def test_runs_and_reports(self):
        hh = _household(savings=[SavingsContribution(name="S", annual_amount=50_000)])
        plan = build_runway_plan(hh, 2033, current_year=CURRENT_YEAR)
        result = run_runway(plan, MarketModel(), n_simulations=500, random_seed=1)
        assert result.n_years == plan.n_years
        assert 0 <= result.success_rate <= 1
        pct = total_spending_percentiles(plan, result)
        assert set(pct) == set(result.spending_percentiles)
        # Pre-retirement consumption is the spending target regardless of markets
        assert np.allclose(pct["P10"][: plan.years_until_retired], plan.annual_spending)

    def test_return_shift_moves_success_rate(self):
        hh = _household(savings=[SavingsContribution(name="S", annual_amount=50_000)])
        plan = build_runway_plan(hh, 2033, current_year=CURRENT_YEAR)
        worse = run_runway(plan, MarketModel(), n_simulations=1000, return_shift=-0.02, random_seed=1)
        better = run_runway(plan, MarketModel(), n_simulations=1000, return_shift=+0.02, random_seed=1)
        assert better.success_rate > worse.success_rate
        assert better.mean_terminal_wealth > worse.mean_terminal_wealth


class TestLognormalReturns:
    def test_geometric_mean_matches_input(self):
        mm = MarketModel(stock_return=0.05, bond_return=0.015, stock_volatility=0.16, bond_volatility=0.06)
        r = generate_returns(200_000, 1, mm, stock_weight=0.6, random_seed=0)
        target = mm.expected_portfolio_return(0.6)
        geo = np.exp(np.mean(np.log1p(r))) - 1
        assert geo == pytest.approx(target, abs=0.002)
        # Simple returns can never be below -100%
        assert r.min() > -1.0


class TestDatedLiabilities:
    def _fees(self, **kw):
        base = dict(name="University fees", liability_type=LiabilityType.DISCRETIONARY_SPENDING,
                    annual_amount=30_000, start_year=2, end_year=6, is_essential=True)
        base.update(kw)
        return Liability(**base)

    def test_is_ongoing_and_household_totals_exclude_dated(self):
        hh = _household()
        hh.liabilities.append(self._fees())
        assert not is_ongoing(hh.liabilities[-1])
        assert hh.total_spending == 70_000
        assert hh.essential_spending == 50_000
        assert [l.name for l in hh.dated_liabilities] == ["University fees"]

    def test_liability_in_year_window_is_end_exclusive(self):
        fees = self._fees()
        assert liability_in_year(fees, 1) == 0
        assert liability_in_year(fees, 2) == 30_000
        assert liability_in_year(fees, 5) == 30_000
        assert liability_in_year(fees, 6) == 0

    def test_fixed_nominal_liability_is_eroded(self):
        fees = self._fees(inflation_linkage=InflationLinkage.NONE)
        assert liability_in_year(fees, 2) == pytest.approx(30_000 / 1.025 ** 3)

    def test_dated_cost_is_drawn_from_portfolio_while_working(self):
        hh = _household(savings=[SavingsContribution(name="S", annual_amount=50_000)])
        base = build_runway_plan(hh, 2033, current_year=CURRENT_YEAR)
        hh.liabilities.append(self._fees())
        plan = build_runway_plan(hh, 2033, current_year=CURRENT_YEAR)
        # Years 2-5 draw the fees (grossed up) against the £50K contribution; other years unchanged
        assert np.allclose(plan.net_spending_by_year[[0, 1, 6]], base.net_spending_by_year[[0, 1, 6]])
        for yr in range(2, 6):
            extra = plan.net_spending_by_year[yr] - base.net_spending_by_year[yr]
            assert extra >= 30_000
            assert extra == pytest.approx(30_000 + plan.tax_by_year[yr], abs=0.1)
        assert np.all(plan.spending_by_year[2:6] == 100_000)
        assert plan.annual_spending == 70_000  # ongoing target unaffected
        assert plan.dated_by_year[3] == 30_000

    def test_dated_cost_after_retirement_raises_the_draw(self):
        hh = _household()
        base = build_runway_plan(hh, 2033, current_year=CURRENT_YEAR)
        hh.liabilities.append(self._fees(start_year=10, end_year=11, is_essential=False))
        plan = build_runway_plan(hh, 2033, current_year=CURRENT_YEAR)
        assert plan.net_spending_by_year[10] > base.net_spending_by_year[10] + 30_000
        assert plan.floor_by_year[10] == pytest.approx(base.floor_by_year[10])  # not essential
        assert np.allclose(plan.net_spending_by_year[11:], base.net_spending_by_year[11:])

    def test_spending_override_only_changes_ongoing(self):
        hh = _household()
        hh.liabilities.append(self._fees())
        plan = build_runway_plan(hh, 2033, current_year=CURRENT_YEAR, annual_spending=80_000)
        assert plan.annual_spending == 80_000
        assert plan.spending_by_year[3] == 110_000
        assert plan.spending_by_year[10] == 80_000

    def test_total_spending_percentiles_include_dated_costs_pre_retirement(self):
        hh = _household(savings=[SavingsContribution(name="S", annual_amount=50_000)])
        hh.liabilities.append(self._fees())
        plan = build_runway_plan(hh, 2033, current_year=CURRENT_YEAR)
        result = run_runway(plan, MarketModel(), n_simulations=200, random_seed=1)
        pct = total_spending_percentiles(plan, result)
        assert pct["P50"][3] == 100_000
        assert pct["P50"][0] == 70_000


class TestSeveranceFloor:
    def test_severance_year_floor_is_reduced_by_the_inflow(self):
        member = Person(name="A", age=54, social_security_annual=12_000, thai_severance_enabled=True,
                        thai_monthly_wage_thb=300_000, thai_employment_start_year=2005, thai_thb_per_gbp=44.0)
        hh = _household(members=[member], savings=[SavingsContribution(name="S", annual_amount=50_000)])
        plan = build_runway_plan(hh, 2033, current_year=CURRENT_YEAR)
        idx = plan.years_until_retired
        without = build_runway_plan(_household(members=[Person(name="A", age=54, social_security_annual=12_000)],
                                               savings=hh.savings_contributions), 2033, current_year=CURRENT_YEAR)
        assert plan.floor_by_year[idx] == pytest.approx(max(0.0, without.floor_by_year[idx] - plan.severance))
        # A path with ample wealth must not register a floor breach in the severance year
        result = run_runway(plan, MarketModel(), n_simulations=300, random_seed=1)
        breach_years = result.time_to_floor_breach[np.isfinite(result.time_to_floor_breach)]
        assert not np.any(breach_years == idx)
