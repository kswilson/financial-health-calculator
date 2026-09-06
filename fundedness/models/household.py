"""Household and person models."""

import datetime
from typing import Optional

from pydantic import BaseModel, Field, model_validator

from fundedness.models.assets import BalanceSheet
from fundedness.models.liabilities import Liability


class Person(BaseModel):
    """An individual person in the household."""

    name: str = Field(..., description="Person's name")
    date_of_birth: Optional[datetime.date] = Field(
        default=None,
        description="Date of birth (used to compute age automatically)",
    )
    age: int = Field(..., ge=0, le=120, description="Current age")
    retirement_age: Optional[int] = Field(
        default=None,
        ge=0,
        le=120,
        description="Expected retirement age (None if already retired)",
    )
    life_expectancy: int = Field(
        default=95,
        ge=0,
        le=120,
        description="Planning life expectancy",
    )
    social_security_age: int = Field(
        default=67,
        ge=62,
        le=70,
        description="Age to claim Social Security",
    )
    social_security_annual: float = Field(
        default=0,
        ge=0,
        description="Expected annual Social Security benefit at claiming age (today's dollars)",
    )
    pension_annual: float = Field(
        default=0,
        ge=0,
        description="Expected annual DB pension benefit",
    )
    pension_start_age: Optional[int] = Field(
        default=60,
        ge=0,
        le=120,
        description="Age when DB pension payments begin",
    )
    pension_inflation_linked: bool = Field(
        default=False,
        description="Whether DB pension increases with inflation (CPI/RPI)",
    )
    is_primary: bool = Field(
        default=True,
        description="Whether this is the primary earner/planner",
    )

    # Thai severance pay (Labour Protection Act)
    thai_severance_enabled: bool = Field(
        default=False,
        description="Whether to include Thai mandatory severance pay",
    )
    thai_monthly_wage_thb: float = Field(
        default=0,
        ge=0,
        description="Monthly wage in Thai Baht",
    )
    thai_employment_start_year: Optional[int] = Field(
        default=None,
        description="Year employment started in Thailand",
    )
    thai_thb_per_gbp: float = Field(
        default=44.0,
        gt=0,
        description="Exchange rate: Thai Baht per GBP",
    )

    def thai_severance_gbp(self, retirement_year: int | None = None) -> float:
        """Calculate Thai severance pay in GBP.

        Args:
            retirement_year: Calendar year of retirement (uses current year if None)

        Returns:
            Severance amount in GBP
        """
        if not self.thai_severance_enabled or not self.thai_employment_start_year:
            return 0.0

        end_year = retirement_year or datetime.date.today().year
        years_of_service = end_year - self.thai_employment_start_year

        # Thai Labour Protection Act Section 118 tiers
        if years_of_service * 365 < 120:
            severance_days = 0
        elif years_of_service < 1:
            severance_days = 30
        elif years_of_service < 3:
            severance_days = 90
        elif years_of_service < 6:
            severance_days = 180
        elif years_of_service < 10:
            severance_days = 240
        elif years_of_service < 20:
            severance_days = 300
        else:
            severance_days = 400

        daily_wage = self.thai_monthly_wage_thb / 30
        severance_thb = daily_wage * severance_days
        return severance_thb / self.thai_thb_per_gbp

    @model_validator(mode="after")
    def validate_ages(self) -> "Person":
        """Validate age relationships and compute age from date_of_birth."""
        if self.date_of_birth is not None:
            today = datetime.date.today()
            self.age = (
                today.year - self.date_of_birth.year
                - ((today.month, today.day) < (self.date_of_birth.month, self.date_of_birth.day))
            )
        if self.retirement_age is not None and self.retirement_age < self.age:
            # Already past retirement age, treat as retired
            self.retirement_age = None
        if self.life_expectancy < self.age:
            raise ValueError("Life expectancy must be greater than current age")
        return self

    @property
    def years_to_retirement(self) -> int:
        """Years until retirement (0 if already retired)."""
        if self.retirement_age is None:
            return 0
        return max(0, self.retirement_age - self.age)

    @property
    def years_in_retirement(self) -> int:
        """Expected years in retirement."""
        retirement_age = self.retirement_age or self.age
        return max(0, self.life_expectancy - retirement_age)

    @property
    def planning_horizon(self) -> int:
        """Total years in planning horizon."""
        return max(0, self.life_expectancy - self.age)


class SavingsContribution(BaseModel):
    """A pre-retirement savings contribution stream."""

    name: str = Field(..., description="Description (e.g. 'SIPP contributions')")
    annual_amount: float = Field(
        ..., ge=0, description="Annual contribution in today's pounds"
    )
    growth_rate: float = Field(
        default=0.0,
        ge=-0.05,
        le=0.10,
        description="Real annual growth rate (e.g. salary increases above inflation)",
    )
    member_name: Optional[str] = Field(
        default=None,
        description="Which household member this belongs to (None = primary)",
    )


class Household(BaseModel):
    """A household unit for financial planning."""

    name: str = Field(
        default="My Household",
        description="Household name",
    )
    members: list[Person] = Field(
        default_factory=list,
        description="Household members",
    )
    balance_sheet: BalanceSheet = Field(
        default_factory=BalanceSheet,
        description="Household balance sheet",
    )
    liabilities: list[Liability] = Field(
        default_factory=list,
        description="Future spending obligations",
    )
    savings_contributions: list[SavingsContribution] = Field(
        default_factory=list,
        description="Pre-retirement savings streams (stop at retirement)",
    )
    state: str = Field(
        default="CA",
        description="State of residence (for tax calculations)",
    )
    filing_status: str = Field(
        default="married_filing_jointly",
        description="Tax filing status",
    )

    @property
    def primary_member(self) -> Optional[Person]:
        """Get the primary household member."""
        for member in self.members:
            if member.is_primary:
                return member
        return self.members[0] if self.members else None

    @property
    def planning_horizon(self) -> int:
        """Planning horizon based on longest-lived member."""
        if not self.members:
            return 30  # Default
        return max(member.planning_horizon for member in self.members)

    @property
    def total_assets(self) -> float:
        """Total asset value."""
        return self.balance_sheet.total_value

    @property
    def ongoing_liabilities(self) -> list[Liability]:
        """Spending that starts now and runs for life (excludes dated one-off costs)."""
        return [l for l in self.liabilities if l.start_year == 0 and l.end_year is None]

    @property
    def dated_liabilities(self) -> list[Liability]:
        """Costs with a start and/or end year (e.g. university fees)."""
        return [l for l in self.liabilities if not (l.start_year == 0 and l.end_year is None)]

    @property
    def essential_spending(self) -> float:
        """Annual essential spending that runs for life (dated one-offs excluded)."""
        return sum(l.annual_amount for l in self.ongoing_liabilities if l.is_essential)

    @property
    def discretionary_spending(self) -> float:
        """Annual discretionary spending that runs for life (dated one-offs excluded)."""
        return sum(l.annual_amount for l in self.ongoing_liabilities if not l.is_essential)

    @property
    def total_spending(self) -> float:
        """Annual ongoing spending target (dated one-offs excluded)."""
        return self.essential_spending + self.discretionary_spending
