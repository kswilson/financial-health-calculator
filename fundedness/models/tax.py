"""Tax model for after-tax calculations (UK tax system with progressive bands)."""

from pydantic import BaseModel, Field

from fundedness.models.assets import AccountType


class TaxModel(BaseModel):
    """UK progressive tax model (2025/26 rates).

    Income tax bands:
        Personal Allowance: £12,570 (tapers £1 for every £2 above £100k)
        Basic rate: 20% (£12,571 – £50,270)
        Higher rate: 40% (£50,271 – £125,140)
        Additional rate: 45% (above £125,140)

    Capital Gains Tax (2025/26):
        Annual exempt amount: £3,000
        Basic rate: 18%
        Higher rate: 24%
    """

    # Income tax bands
    personal_allowance: float = Field(
        default=12_570,
        ge=0,
        description="Income tax personal allowance",
    )
    basic_rate_limit: float = Field(
        default=50_270,
        ge=0,
        description="Upper limit of basic rate band",
    )
    higher_rate_limit: float = Field(
        default=125_140,
        ge=0,
        description="Upper limit of higher rate band",
    )
    basic_rate: float = Field(
        default=0.20,
        ge=0,
        le=1,
        description="Basic rate of income tax",
    )
    higher_rate: float = Field(
        default=0.40,
        ge=0,
        le=1,
        description="Higher rate of income tax",
    )
    additional_rate: float = Field(
        default=0.45,
        ge=0,
        le=1,
        description="Additional rate of income tax",
    )

    # Other taxable income (State Pension, DB pension, salary)
    # This determines where in the bands SIPP withdrawals start
    other_income: float = Field(
        default=0,
        ge=0,
        description="Annual income from other sources (State Pension, DB pension, salary)",
    )

    # Capital gains tax
    cgt_annual_exempt: float = Field(
        default=3_000,
        ge=0,
        description="CGT annual exempt amount",
    )
    cgt_basic_rate: float = Field(
        default=0.18,
        ge=0,
        le=1,
        description="CGT rate for basic rate taxpayers",
    )
    cgt_higher_rate: float = Field(
        default=0.24,
        ge=0,
        le=1,
        description="CGT rate for higher rate taxpayers",
    )

    # Cost basis assumptions
    default_cost_basis_ratio: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Default cost basis as fraction of value (for unrealised gains)",
    )

    def _effective_personal_allowance(self, total_income: float) -> float:
        """Personal allowance tapers £1 for every £2 above £100k."""
        if total_income <= 100_000:
            return self.personal_allowance
        taper = (total_income - 100_000) / 2
        return max(0, self.personal_allowance - taper)

    def compute_income_tax(self, withdrawal: float) -> float:
        """Compute income tax on a withdrawal given other income.

        Models UK progressive bands: the withdrawal is stacked on top
        of other_income to determine which bands it falls into.

        Args:
            withdrawal: Amount being withdrawn (e.g. from SIPP)

        Returns:
            Total income tax due on the withdrawal
        """
        if withdrawal <= 0:
            return 0.0

        total_income = self.other_income + withdrawal
        pa = self._effective_personal_allowance(total_income)

        # Build tax bands: (threshold, rate)
        bands = [
            (pa, 0.0),
            (self.basic_rate_limit, self.basic_rate),
            (self.higher_rate_limit, self.higher_rate),
            (float("inf"), self.additional_rate),
        ]

        # Tax on total income
        def _tax_on(income: float) -> float:
            tax = 0.0
            prev = 0.0
            for threshold, rate in bands:
                if income <= prev:
                    break
                taxable = min(income, threshold) - prev
                if taxable > 0:
                    tax += taxable * rate
                prev = threshold
            return tax

        # Tax attributable to the withdrawal = tax on total - tax on other income alone
        tax_on_total = _tax_on(total_income)
        tax_on_other = _tax_on(self.other_income)
        return tax_on_total - tax_on_other

    def compute_cgt(self, gains: float) -> float:
        """Compute capital gains tax on realised gains.

        Applies the annual exempt amount, then taxes at basic or higher
        rate depending on whether total income uses up the basic band.

        Args:
            gains: Realised capital gains

        Returns:
            CGT due
        """
        if gains <= 0:
            return 0.0

        taxable_gains = max(0, gains - self.cgt_annual_exempt)
        if taxable_gains <= 0:
            return 0.0

        # How much basic rate band is left after other income?
        basic_band_remaining = max(0, self.basic_rate_limit - self.other_income)

        # Gains taxed at basic rate (up to remaining band)
        gains_at_basic = min(taxable_gains, basic_band_remaining)
        gains_at_higher = taxable_gains - gains_at_basic

        return gains_at_basic * self.cgt_basic_rate + gains_at_higher * self.cgt_higher_rate

    @property
    def total_ordinary_rate(self) -> float:
        """Marginal income tax rate given other income (for compatibility)."""
        # Return the marginal rate at the current other_income level
        pa = self._effective_personal_allowance(self.other_income)
        if self.other_income <= pa:
            return 0.0
        elif self.other_income <= self.basic_rate_limit:
            return self.basic_rate
        elif self.other_income <= self.higher_rate_limit:
            return self.higher_rate
        else:
            return self.additional_rate

    @property
    def total_ltcg_rate(self) -> float:
        """Marginal CGT rate given other income (for compatibility)."""
        if self.other_income <= self.basic_rate_limit:
            return self.cgt_basic_rate
        return self.cgt_higher_rate

    def get_effective_tax_rate(
        self,
        account_type: AccountType,
        cost_basis_ratio: float | None = None,
        amount: float = 0,
    ) -> float:
        """Get the effective average tax rate for withdrawals from an account.

        For SIPP/tax-deferred accounts, computes progressive income tax
        on the withdrawal stacked on top of other income.

        For GIA/taxable accounts, computes CGT on the gains portion
        with the annual exempt amount.

        Args:
            account_type: Type of account
            cost_basis_ratio: Cost basis as fraction of value (for taxable accounts)
            amount: Withdrawal/asset amount (used for progressive band calculation)

        Returns:
            Effective average tax rate as decimal (0-1)
        """
        match account_type:
            case AccountType.TAX_EXEMPT | AccountType.ISA | AccountType.HSA:
                return 0.0

            case AccountType.TAX_DEFERRED | AccountType.SIPP:
                if amount <= 0:
                    return self.total_ordinary_rate
                tax = self.compute_income_tax(amount)
                return tax / amount

            case AccountType.TAXABLE | AccountType.GENERAL:
                if cost_basis_ratio is None:
                    cost_basis_ratio = self.default_cost_basis_ratio

                gains = amount * (1 - cost_basis_ratio) if amount > 0 else 0
                if gains <= 0 or amount <= 0:
                    return (1 - cost_basis_ratio) * self.total_ltcg_rate

                cgt = self.compute_cgt(gains)
                return cgt / amount

            case _:
                return 0.0

    def get_haircut_by_account_type(self) -> dict[AccountType, float]:
        """Get tax haircut factors by account type.

        Returns:
            Dictionary mapping account type to (1 - tax_rate)
        """
        return {
            at: 1 - self.get_effective_tax_rate(at)
            for at in AccountType
        }


def build_tax_models_for_household(
    household: "Household",
    base_tax_model: TaxModel,
) -> dict[str, TaxModel]:
    """Build per-person TaxModels from household members.

    Each person gets their own TaxModel with other_income set to their
    State Pension + DB pension. Band parameters are shared (UK bands
    are the same for everyone).

    Args:
        household: Household with members
        base_tax_model: Base tax model with band parameters

    Returns:
        Dict mapping person name to their TaxModel
    """
    from fundedness.models.household import Household  # noqa: F811

    tax_models: dict[str, TaxModel] = {}
    for member in household.members:
        other_income = member.social_security_annual + member.pension_annual
        tax_models[member.name] = base_tax_model.model_copy(
            update={"other_income": other_income}
        )
    return tax_models
