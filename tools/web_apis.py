from __future__ import annotations

import hashlib
from typing import Any, Dict, List

from ..schemas import ToolParameter, ToolSchema
from ..toolface import ToolFace


_FLIGHTS = [
    {"id": "FL-AA101", "from": "ORD", "to": "SFO", "depart": "2026-06-12T07:30",
     "arrive": "2026-06-12T10:05", "price_usd": 287, "carrier": "American"},
    {"id": "FL-UA205", "from": "ORD", "to": "SFO", "depart": "2026-06-12T09:50",
     "arrive": "2026-06-12T12:25", "price_usd": 312, "carrier": "United"},
    {"id": "FL-DL319", "from": "ORD", "to": "JFK", "depart": "2026-06-12T08:10",
     "arrive": "2026-06-12T11:30", "price_usd": 198, "carrier": "Delta"},
]

_HOTELS = [
    {"id": "HT-001", "city": "SFO", "name": "Mission Bay Suites",
     "nightly_usd": 189, "rating": 4.4},
    {"id": "HT-002", "city": "SFO", "name": "Embarcadero Inn",
     "nightly_usd": 245, "rating": 4.6},
    {"id": "HT-003", "city": "JFK", "name": "Brooklyn Loft",
     "nightly_usd": 175, "rating": 4.3},
]

_PRODUCTS = [
    {"id": "P-1001", "name": "Wireless headphones", "price_usd":  79.0, "in_stock": True},
    {"id": "P-1002", "name": "USB-C charger 65W",   "price_usd":  29.5, "in_stock": True},
    {"id": "P-1003", "name": "Mechanical keyboard", "price_usd": 149.0, "in_stock": False},
]

_GAS_STATIONS = {
    "61820": [
        {"name": "Shell - Green St",   "regular_usd_gal": 3.49},
        {"name": "BP - Neil St",       "regular_usd_gal": 3.39},
        {"name": "Costco - N Mattis",  "regular_usd_gal": 3.21},
    ],
    "94103": [
        {"name": "Chevron - 5th",      "regular_usd_gal": 4.89},
        {"name": "76 - Bryant",        "regular_usd_gal": 4.79},
    ],
}


def search_flights(origin: str, destination: str, date: str) -> List[Dict[str, Any]]:
    origin = origin.upper(); destination = destination.upper()
    return [f for f in _FLIGHTS
            if f["from"] == origin and f["to"] == destination
            and f["depart"].startswith(date)]


def book_flight(flight_id: str, passenger_name: str) -> Dict[str, Any]:
    matches = [f for f in _FLIGHTS if f["id"] == flight_id]
    if not matches:
        raise ValueError(f"unknown flight_id: {flight_id}")
    f = matches[0]
    pnr = hashlib.md5(f"{flight_id}|{passenger_name}".encode()).hexdigest()[:6].upper()
    return {
        "pnr": pnr, "flight": f, "passenger": passenger_name,
        "status": "CONFIRMED", "total_usd": f["price_usd"],
    }


def search_hotels(city: str, max_price: float = 1000.0) -> List[Dict[str, Any]]:
    city = city.upper()
    return [h for h in _HOTELS if h["city"] == city and h["nightly_usd"] <= max_price]


def book_hotel(hotel_id: str, nights: int, guest_name: str) -> Dict[str, Any]:
    matches = [h for h in _HOTELS if h["id"] == hotel_id]
    if not matches:
        raise ValueError(f"unknown hotel_id: {hotel_id}")
    if nights <= 0:
        raise ValueError(f"nights must be positive, got {nights}")
    h = matches[0]
    res_id = hashlib.md5(f"{hotel_id}|{guest_name}|{nights}".encode()).hexdigest()[:6].upper()
    return {
        "reservation_id": res_id, "hotel": h, "guest": guest_name,
        "nights": nights, "total_usd": h["nightly_usd"] * nights,
        "status": "CONFIRMED",
    }


def search_products(query: str, in_stock_only: bool = True) -> List[Dict[str, Any]]:
    q = query.lower()
    out = []
    for p in _PRODUCTS:
        if q in p["name"].lower():
            if in_stock_only and not p["in_stock"]:
                continue
            out.append(p)
    return out


def place_order(product_id: str, quantity: int = 1) -> Dict[str, Any]:
    matches = [p for p in _PRODUCTS if p["id"] == product_id]
    if not matches:
        raise ValueError(f"unknown product_id: {product_id}")
    p = matches[0]
    if not p["in_stock"]:
        raise ValueError(f"{p['name']} is out of stock")
    if quantity <= 0:
        raise ValueError(f"quantity must be positive, got {quantity}")
    return {
        "order_id": hashlib.md5(f"{product_id}|{quantity}".encode()).hexdigest()[:6].upper(),
        "product": p, "quantity": quantity, "total_usd": p["price_usd"] * quantity,
        "status": "PLACED",
    }


def get_gas_prices(zip_code: str) -> List[Dict[str, Any]]:
    if zip_code not in _GAS_STATIONS:
        return []
    return _GAS_STATIONS[zip_code]


def register(tf: ToolFace) -> None:
    tf.register(
        ToolSchema(
            id="search_flights",
            name="Search flights",
            description="Find available flights between two airports on a given date.",
            category="travel",
            source="toolbench",
            parameters=[
                ToolParameter("origin", "string", required=True,
                              description="Departure IATA code (e.g., 'ORD')."),
                ToolParameter("destination", "string", required=True,
                              description="Arrival IATA code (e.g., 'SFO')."),
                ToolParameter("date", "string", required=True,
                              description="Departure date in YYYY-MM-DD."),
            ],
            returns="List of flight records with id, from/to, depart, arrive, price_usd, carrier.",
        ),
        search_flights,
    )
    tf.register(
        ToolSchema(
            id="book_flight",
            name="Book flight",
            description="Confirm a booking for a specific flight under a passenger name.",
            category="travel",
            source="toolbench",
            parameters=[
                ToolParameter("flight_id", "string", required=True,
                              description="Flight id returned by search_flights."),
                ToolParameter("passenger_name", "string", required=True,
                              description="Full name of the passenger."),
            ],
            returns="{pnr, flight, passenger, status, total_usd}",
        ),
        book_flight,
    )
    tf.register(
        ToolSchema(
            id="search_hotels",
            name="Search hotels",
            description="Find hotels in a city under an optional price ceiling.",
            category="travel",
            source="toolbench",
            parameters=[
                ToolParameter("city", "string", required=True,
                              description="City airport code (e.g., 'SFO')."),
                ToolParameter("max_price", "number", required=False, default=1000.0,
                              description="Maximum nightly price in USD."),
            ],
            returns="List of hotels with id, city, name, nightly_usd, rating.",
        ),
        search_hotels,
    )
    tf.register(
        ToolSchema(
            id="book_hotel",
            name="Book hotel",
            description="Reserve a hotel room for N nights under a guest name.",
            category="travel",
            source="toolbench",
            parameters=[
                ToolParameter("hotel_id", "string", required=True,
                              description="Hotel id returned by search_hotels."),
                ToolParameter("nights", "integer", required=True,
                              description="Number of nights to stay."),
                ToolParameter("guest_name", "string", required=True,
                              description="Full name of the primary guest."),
            ],
            returns="{reservation_id, hotel, guest, nights, total_usd, status}",
        ),
        book_hotel,
    )

    tf.register(
        ToolSchema(
            id="search_products",
            name="Search products",
            description="Free-text search the product catalogue.",
            category="ecommerce",
            source="toolbench",
            parameters=[
                ToolParameter("query", "string", required=True,
                              description="Search term — name or keyword."),
                ToolParameter("in_stock_only", "boolean", required=False, default=True,
                              description="If true, filter out unavailable items."),
            ],
            returns="List of products with id, name, price_usd, in_stock.",
        ),
        search_products,
    )
    tf.register(
        ToolSchema(
            id="place_order",
            name="Place order",
            description="Place an order for a known product id.",
            category="ecommerce",
            source="toolbench",
            parameters=[
                ToolParameter("product_id", "string", required=True,
                              description="Product id returned by search_products."),
                ToolParameter("quantity", "integer", required=False, default=1,
                              description="Number of units to order."),
            ],
            returns="{order_id, product, quantity, total_usd, status}",
        ),
        place_order,
    )

    tf.register(
        ToolSchema(
            id="get_gas_prices",
            name="Get local gas prices",
            description="Look up current regular-gas prices near a US ZIP code.",
            category="local",
            source="manual",
            parameters=[
                ToolParameter("zip_code", "string", required=True,
                              description="5-digit US ZIP code."),
            ],
            returns="List of {name, regular_usd_gal}.",
        ),
        get_gas_prices,
    )
