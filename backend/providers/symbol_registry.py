"""
symbol_registry.py
-------------------
Central symbol registry for TradeSense V2.1.

Manages canonical NSE ticker symbols, provider-specific mappings,
and accounts for corporate actions (e.g. demergers, mergers, and ticker renames)
such as the October 2025 Tata Motors restructuring and REC Ltd. exchange ticker conventions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class SymbolInfo:
    """Metadata regarding a ticker's canonical identity and provider aliases."""
    canonical_symbol: str
    company_name: str
    sector: str
    industry: str
    exchange: str = "NSE"
    # Mapping of provider identifier -> provider-specific ticker symbol
    provider_symbols: dict[str, str] = field(default_factory=dict)
    corporate_action: Optional[str] = None
    corporate_action_notes: Optional[str] = None
    is_economically_continuous: bool = True


class CentralSymbolRegistry:
    """
    Single source of truth for symbol resolution across all market data providers.
    Prevents scattered ticker replacements across the codebase.
    """

    def __init__(self) -> None:
        self._registry: dict[str, SymbolInfo] = {}
        self._reverse_map: dict[str, dict[str, str]] = {}  # provider -> {provider_sym: canonical}
        self._initialize_known_corporate_actions()

    def _initialize_known_corporate_actions(self) -> None:
        """Register documented ticker mappings and corporate action histories."""
        # 1. REC Ltd:
        # NSE listed symbol is RECLTD, while legacy/short representations use REC.NS.
        self.register(SymbolInfo(
            canonical_symbol="REC.NS",
            company_name="REC Ltd.",
            sector="Financial Services",
            industry="NBFC",
            provider_symbols={
                "yfinance": "RECLTD.NS",
                "backup_api": "RECLTD.NS",
                "disk_cache": "REC_NS",
            },
            corporate_action="TICKER_CONVENTION",
            corporate_action_notes="Listed on NSE as RECLTD. Yahoo Finance indexes REC as RECLTD.NS.",
            is_economically_continuous=True,
        ))

        # 2. Tata Motors Restructuring (October 2025):
        # Demerger into two distinct listed entities:
        # - TMPV: Tata Motors Passenger Vehicles Limited (passenger cars & EV)
        # - TMCV: Tata Motors Limited, commercial vehicles business.
        # On Yahoo Finance post-demerger, TATAMOTORS.NS trading ceased or split into TMPV.NS.
        self.register(SymbolInfo(
            canonical_symbol="TATAMOTORS.NS",
            company_name="Tata Motors Ltd.",
            sector="Automobile",
            industry="Passenger & Commercial Vehicles",
            provider_symbols={
                "yfinance": "TMPV.NS",
                "backup_api": "TMPV.NS",
                "disk_cache": "TATAMOTORS_NS",
            },
            corporate_action="DEMERGER_OCT_2025",
            corporate_action_notes=(
                "Demerged in October 2025 into TMPV (Passenger Vehicles) and TMCV (Commercial Vehicles). "
                "TMPV provides operational and price series continuation for the passenger car segment."
            ),
            is_economically_continuous=False,
        ))

        # 3. LTIMindtree Ltd:
        # Formed by the merger of L&T Infotech (LTI) and Mindtree.
        # Note: LTTS (L&T Technology Services) is a SEPARATE company and must NEVER be mapped as LTIM!
        self.register(SymbolInfo(
            canonical_symbol="LTIM.NS",
            company_name="LTIMindtree Ltd.",
            sector="Information Technology",
            industry="IT Consulting & Software",
            provider_symbols={
                "yfinance": "LTIM.NS",
                "backup_api": "LTIM.NS",
                "disk_cache": "LTIM_NS",
            },
            corporate_action="MERGER_LTI_MINDTREE",
            corporate_action_notes="LTIMindtree merged entity. Distinct from LTTS (L&T Tech Services).",
            is_economically_continuous=True,
        ))

    def register(self, info: SymbolInfo) -> None:
        """Register or update a symbol's metadata and provider aliases."""
        self._registry[info.canonical_symbol] = info
        for provider, prov_sym in info.provider_symbols.items():
            if provider not in self._reverse_map:
                self._reverse_map[provider] = {}
            self._reverse_map[provider][prov_sym] = info.canonical_symbol

    def get_provider_symbol(self, canonical_symbol: str, provider: str = "yfinance") -> str:
        """
        Resolve canonical symbol to provider-specific query symbol.
        Defaults to canonical_symbol if no specific override is registered.
        """
        info = self._registry.get(canonical_symbol)
        if info and provider in info.provider_symbols:
            return info.provider_symbols[provider]
        return canonical_symbol

    def get_canonical_symbol(self, provider_symbol: str, provider: str = "yfinance") -> str:
        """Resolve a provider-specific ticker back to its canonical symbol."""
        if provider in self._reverse_map and provider_symbol in self._reverse_map[provider]:
            return self._reverse_map[provider][provider_symbol]
        return provider_symbol

    def get_symbol_info(self, canonical_symbol: str) -> Optional[SymbolInfo]:
        """Retrieve full metadata for a canonical symbol."""
        return self._registry.get(canonical_symbol)

    def is_corporate_action(self, symbol: str) -> bool:
        """Check if symbol has a documented corporate action restructuring."""
        info = self._registry.get(symbol)
        return bool(info and info.corporate_action)


# Global singleton instance
symbol_registry = CentralSymbolRegistry()
