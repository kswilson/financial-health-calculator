"""CEFR (Certainty-Equivalent Funded Ratio) calculation engine."""

from dataclasses import dataclass, field

from fundedness.liabilities import calculate_annuity_pv, calculate_total_liability_pv
from fundedness.liquidity import get_liquidity_factor
from fundedness.models.assets import AccountType, Asset, BalanceSheet
from fundedness.models.household import Household
from fundedness.models.liabilities import Liability
from fundedness.models.tax import TaxModel
from fundedness.risk import get_reliability_factor


@dataclass
class AssetHaircutDetail:
    """Detailed haircut breakdown for a single asset."""

    asset: Asset
    gross_value: float
    tax_rate: float
    after_tax_value: float
    liquidity_factor: float
    after_liquidity_value: float
    reliability_factor: float
    net_value: float

    @property
    def total_haircut(self) -> float:
        """Total haircut as decimal (1 - net/gross)."""
        if self.gross_value == 0:
            return 0.0
        return 1 - (self.net_value / self.gross_value)

    @property
    def tax_haircut(self) -> float:
        """Tax haircut amount in dollars."""
        return self.gross_value - self.after_tax_value

    @property
    def liquidity_haircut(self) -> float:
        """Liquidity haircut amount in dollars."""
        return self.after_tax_value - self.after_liquidity_value

    @property
    def reliability_haircut(self) -> float:
        """Reliability haircut amount in dollars."""
        return self.after_liquidity_value - self.net_value


@dataclass
class CEFRResult:
    """Complete CEFR calculation result with breakdown."""

    # Main ratio
    cefr: float

    # Numerator components
    gross_assets: float
    total_tax_haircut: float
    total_liquidity_haircut: float
    total_reliability_haircut: float
    net_assets: float

    # Denominator
    liability_pv: float
    pension_income_pv: float = 0.0  # PV of pension income that offsets liabilities
    net_liability_pv: float = 0.0  # liability_pv - pension_income_pv

    # Detailed breakdowns
    asset_details: list[AssetHaircutDetail] = field(default_factory=list)

    @property
    def total_haircut(self) -> float:
        """Total haircut amount."""
        return self.total_tax_haircut + self.total_liquidity_haircut + self.total_reliability_haircut

    @property
    def haircut_percentage(self) -> float:
        """Total haircut as percentage of gross assets."""
        if self.gross_assets == 0:
            return 0.0
        return self.total_haircut / self.gross_assets

    @property
    def is_funded(self) -> bool:
        """Whether CEFR >= 1.0 (fully funded)."""
        return self.cefr >= 1.0

    @property
    def funding_gap(self) -> float:
        """Dollar gap if underfunded (positive = gap, negative = surplus)."""
        return self.net_liability_pv - self.net_assets

    def get_interpretation(self) -> str:
        """Get a human-readable interpretation of the CEFR."""
        if self.cefr >= 2.0:
            return "Excellent: Very well-funded with significant buffer"
        elif self.cefr >= 1.5:
            return "Strong: Well-funded with comfortable margin"
        elif self.cefr >= 1.0:
            return "Adequate: Fully funded but limited cushion"
        elif self.cefr >= 0.8:
            return "Marginal: Slightly underfunded, minor adjustments needed"
        elif self.cefr >= 0.5:
            return "Concerning: Significantly underfunded, action required"
        else:
            return "Critical: Severely underfunded, major changes needed"


def compute_asset_haircuts(
    asset: Asset,
    tax_model: TaxModel,
) -> AssetHaircutDetail:
    """Compute all haircuts for a single asset.

    Args:
        asset: The asset to analyze
        tax_model: Tax rate assumptions

    Returns:
        Detailed haircut breakdown
    """
    gross_value = asset.value

    # Step 1: Tax haircut
    cost_basis_ratio = None
    if asset.cost_basis is not None and asset.value > 0:
        cost_basis_ratio = asset.cost_basis / asset.value

    tax_rate = tax_model.get_effective_tax_rate(
        account_type=asset.account_type,
        cost_basis_ratio=cost_basis_ratio,
        amount=gross_value,
    )
    after_tax_value = gross_value * (1 - tax_rate)

    # Step 2: Liquidity haircut
    liquidity_factor = get_liquidity_factor(asset.liquidity_class)
    after_liquidity_value = after_tax_value * liquidity_factor

    # Step 3: Reliability haircut
    reliability_factor = get_reliability_factor(
        concentration_level=asset.concentration_level,
        asset_class=asset.asset_class,
    )
    net_value = after_liquidity_value * reliability_factor

    return AssetHaircutDetail(
        asset=asset,
        gross_value=gross_value,
        tax_rate=tax_rate,
        after_tax_value=after_tax_value,
        liquidity_factor=liquidity_factor,
        after_liquidity_value=after_liquidity_value,
        reliability_factor=reliability_factor,
        net_value=net_value,
    )


def calculate_pension_income_pv(
    household: Household,
    planning_horizon: int,
    real_discount_rate: float = 0.02,
    base_inflation: float = 0.025,
) -> float:
    """Calculate the present value of future pension income streams.

    Includes state pension and defined-benefit pension for all members.

    Args:
        household: Household with members
        planning_horizon: Years to project
        real_discount_rate: Real discount rate
        base_inflation: Base inflation assumption

    Returns:
        Total PV of pension income
    """
    total_pv = 0.0
    primary_age = household.primary_member.age if household.primary_member else 65

    for member in household.members:
        # State pension: inflation-linked (constant in real terms)
        sp_age = getattr(member, "social_security_age", None) or 67
        sp_annual = member.social_security_annual
        if sp_annual > 0:
            years_until_sp = max(0, sp_age - member.age)
            years_receiving_sp = max(0, planning_horizon - years_until_sp)
            if years_receiving_sp > 0:
                total_pv += calculate_annuity_pv(
                    annual_payment=sp_annual,
                    n_years=years_receiving_sp,
                    discount_rate=real_discount_rate,
                    growth_rate=0.0,  # CPI-linked = constant in real terms
                    start_year=years_until_sp,
                )

        # DB pension
        db_annual = member.pension_annual
        db_start_age = getattr(member, "pension_start_age", None) or 60
        db_linked = getattr(member, "pension_inflation_linked", False)
        if db_annual > 0:
            years_until_db = max(0, db_start_age - member.age)
            years_receiving_db = max(0, planning_horizon - years_until_db)
            if years_receiving_db > 0:
                # If not inflation-linked, real value erodes
                growth_rate = 0.0 if db_linked else -base_inflation
                total_pv += calculate_annuity_pv(
                    annual_payment=db_annual,
                    n_years=years_receiving_db,
                    discount_rate=real_discount_rate,
                    growth_rate=growth_rate,
                    start_year=years_until_db,
                )

    return total_pv


def compute_cefr(
    household: Household | None = None,
    balance_sheet: BalanceSheet | None = None,
    liabilities: list[Liability] | None = None,
    tax_model: TaxModel | None = None,
    tax_models: dict[str, TaxModel] | None = None,
    planning_horizon: int | None = None,
    real_discount_rate: float = 0.02,
    base_inflation: float = 0.025,
) -> CEFRResult:
    """Compute the Certainty-Equivalent Funded Ratio (CEFR).

    CEFR = Σ(Asset × (1-τ) × λ × ρ) / PV(Liabilities)

    Where:
        τ = tax rate
        λ = liquidity factor
        ρ = reliability factor

    Args:
        household: Complete household model (alternative to separate components)
        balance_sheet: Asset holdings (if household not provided)
        liabilities: Future spending obligations (if household not provided)
        tax_model: Tax rate assumptions for single-person (defaults to TaxModel())
        tax_models: Per-person tax models keyed by person name (for couples)
        planning_horizon: Years to plan for (defaults to household horizon or 30)
        real_discount_rate: Real discount rate for liability PV
        base_inflation: Base inflation assumption

    Returns:
        CEFRResult with complete breakdown
    """
    # Extract components from household or use provided values
    if household is not None:
        balance_sheet = household.balance_sheet
        liabilities = household.liabilities
        if planning_horizon is None:
            planning_horizon = household.planning_horizon
    else:
        if balance_sheet is None:
            balance_sheet = BalanceSheet()
        if liabilities is None:
            liabilities = []

    if planning_horizon is None:
        planning_horizon = 30

    if tax_model is None:
        tax_model = TaxModel()

    # Determine primary owner name for assets with owner=None
    primary_name = None
    if household and household.primary_member:
        primary_name = household.primary_member.name

    # Build per-person tax models if not provided but household has multiple members
    if tax_models is None and household and len(household.members) > 1:
        from fundedness.models.tax import build_tax_models_for_household
        tax_models = build_tax_models_for_household(household, tax_model)

    # Compute asset haircuts with cumulative income stacking per owner.
    # Income-taxed assets (SIPP, TAX_DEFERRED) are grouped by owner so each
    # person's withdrawals stack within their own tax bands independently.
    income_taxed = {AccountType.TAX_DEFERRED, AccountType.SIPP}

    def _get_owner(asset: Asset) -> str:
        return asset.owner or primary_name or "_default"

    if tax_models:
        # Multi-person: group by owner, stack independently per person
        from collections import defaultdict
        income_by_owner: dict[str, list[Asset]] = defaultdict(list)
        other_by_owner: dict[str, list[Asset]] = defaultdict(list)

        for asset in balance_sheet.assets:
            owner = _get_owner(asset)
            if asset.account_type in income_taxed:
                income_by_owner[owner].append(asset)
            else:
                other_by_owner[owner].append(asset)

        asset_details = []

        # Process income-taxed assets per owner with cumulative stacking
        for owner, assets in income_by_owner.items():
            owner_tm = tax_models.get(owner, tax_model).model_copy()
            for asset in assets:
                detail = compute_asset_haircuts(asset, owner_tm)
                asset_details.append(detail)
                owner_tm = owner_tm.model_copy(
                    update={"other_income": owner_tm.other_income + asset.value}
                )

        # Process non-income-taxed assets per owner (independent CGT allowances)
        for owner, assets in other_by_owner.items():
            owner_tm = tax_models.get(owner, tax_model)
            for asset in assets:
                detail = compute_asset_haircuts(asset, owner_tm)
                asset_details.append(detail)
    else:
        # Single-person: original cumulative logic
        income_assets = [a for a in balance_sheet.assets if a.account_type in income_taxed]
        other_assets = [a for a in balance_sheet.assets if a.account_type not in income_taxed]

        asset_details = []
        cumulative_tax_model = tax_model.model_copy()

        for asset in income_assets:
            detail = compute_asset_haircuts(asset, cumulative_tax_model)
            asset_details.append(detail)
            cumulative_tax_model = cumulative_tax_model.model_copy(
                update={"other_income": cumulative_tax_model.other_income + asset.value}
            )

        for asset in other_assets:
            detail = compute_asset_haircuts(asset, tax_model)
            asset_details.append(detail)

    # Aggregate numerator
    gross_assets = sum(d.gross_value for d in asset_details)
    total_tax_haircut = sum(d.tax_haircut for d in asset_details)
    total_liquidity_haircut = sum(d.liquidity_haircut for d in asset_details)
    total_reliability_haircut = sum(d.reliability_haircut for d in asset_details)
    net_assets = sum(d.net_value for d in asset_details)

    # Compute liability PV (denominator)
    liability_pv, _ = calculate_total_liability_pv(
        liabilities=liabilities,
        planning_horizon=planning_horizon,
        real_discount_rate=real_discount_rate,
        base_inflation=base_inflation,
    )

    # Offset pension income against liabilities
    pension_income_pv = 0.0
    if household is not None:
        pension_income_pv = calculate_pension_income_pv(
            household=household,
            planning_horizon=planning_horizon,
            real_discount_rate=real_discount_rate,
            base_inflation=base_inflation,
        )
    net_liability_pv = max(0, liability_pv - pension_income_pv)

    # Calculate CEFR using net liabilities (after pension income offset)
    if net_liability_pv == 0:
        cefr = float("inf") if net_assets > 0 else 0.0
    else:
        cefr = net_assets / net_liability_pv

    return CEFRResult(
        cefr=cefr,
        gross_assets=gross_assets,
        total_tax_haircut=total_tax_haircut,
        total_liquidity_haircut=total_liquidity_haircut,
        total_reliability_haircut=total_reliability_haircut,
        net_assets=net_assets,
        liability_pv=liability_pv,
        pension_income_pv=pension_income_pv,
        net_liability_pv=net_liability_pv,
        asset_details=asset_details,
    )
