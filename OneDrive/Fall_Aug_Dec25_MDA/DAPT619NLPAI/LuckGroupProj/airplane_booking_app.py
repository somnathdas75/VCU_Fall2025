
import sqlite3
import datetime as dt
from pathlib import Path
from typing import List, Tuple, Optional

import streamlit as st
import pandas as pd

DB_PATH = "airplane_bookings.db"

# -----------------------------
# Database helpers
# -----------------------------

def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    with get_conn() as conn:
        cur = conn.cursor()
        # Flights table: one row per flight date
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS flights (
                flight_date TEXT PRIMARY KEY
            );
            """
        )
        # Bookings table: one row per seat reservation
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS bookings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                flight_date TEXT NOT NULL,
                seat_number INTEGER NOT NULL,
                name TEXT NOT NULL,
                age INTEGER NOT NULL,
                dob TEXT NOT NULL,
                price REAL NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE (flight_date, seat_number)
            );
            """
        )
        conn.commit()

def seed_flights(days: int = 90):
    """Create flights for the next `days` days (including today)."""
    today = dt.date.today()
    dates = [(today + dt.timedelta(days=i)).isoformat() for i in range(days)]
    with get_conn() as conn:
        cur = conn.cursor()
        cur.executemany(
            "INSERT OR IGNORE INTO flights (flight_date) VALUES (?)",
            [(d,) for d in dates],
        )
        conn.commit()

def list_flight_dates() -> List[str]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT flight_date FROM flights ORDER BY flight_date ASC")
        return [row[0] for row in cur.fetchall()]

def booked_seats_for_date(flight_date: str) -> List[int]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT seat_number FROM bookings WHERE flight_date = ? ORDER BY seat_number",
            (flight_date,),
        )
        return [int(r[0]) for r in cur.fetchall()]

def fetch_bookings(search_name: Optional[str] = None, flight_date: Optional[str] = None) -> pd.DataFrame:
    query = "SELECT id, flight_date, seat_number, name, age, dob, price, created_at FROM bookings"
    params = []
    clauses = []
    if search_name:
        clauses.append("LOWER(name) LIKE ?")
        params.append(f"%{search_name.lower()}%")
    if flight_date:
        clauses.append("flight_date = ?")
        params.append(flight_date)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY flight_date, seat_number"
    with get_conn() as conn:
        return pd.read_sql_query(query, conn, params=params)

def insert_booking(flight_date: str, seat_number: int, name: str, age: int, dob: str, price: float) -> Tuple[bool, str]:
    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO bookings (flight_date, seat_number, name, age, dob, price)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (flight_date, seat_number, name, age, dob, price),
            )
            conn.commit()
        return True, "Booking confirmed."
    except sqlite3.IntegrityError as e:
        # Likely a unique constraint violation (seat already booked)
        return False, "That seat was just booked by someone else. Please choose another seat."
    except Exception as e:
        return False, f"Failed to book seat: {e}"

def cancel_booking(booking_id: int) -> Tuple[bool, str]:
    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM bookings WHERE id = ?", (booking_id,))
            conn.commit()
        return True, "Booking cancelled."
    except Exception as e:
        return False, f"Failed to cancel booking: {e}"

# -----------------------------
# UI helpers
# -----------------------------

def seat_label(n: int) -> str:
    """Format seat label (e.g., 1 -> '01')."""
    return f"{n:02d}"

def render_seat_map(flight_date: str, seats_per_row: int = 5, total_seats: int = 50):
    """Render a seat map as a grid of buttons. Returns the seat number selected (or None)."""
    st.subheader("Seat Map")
    booked = set(booked_seats_for_date(flight_date))
    selected = None

    rows = (total_seats + seats_per_row - 1) // seats_per_row
    for r in range(rows):
        cols = st.columns(seats_per_row, gap="small")
        for c in range(seats_per_row):
            seat_number = r * seats_per_row + c + 1
            if seat_number > total_seats:
                # Empty placeholder for grid alignment
                cols[c].markdown("&nbsp;")
                continue

            is_booked = seat_number in booked
            label = seat_label(seat_number)

            # Use different styles to indicate availability
            container = cols[c].container(border=True)
            if is_booked:
                container.button(
                    f"💺 {label} (Booked)",
                    key=f"seat_btn_{seat_number}_{flight_date}_booked",
                    disabled=True,
                    help="This seat is already booked."
                )
            else:
                if container.button(
                    f"🟢 {label} (Free)",
                    key=f"seat_btn_{seat_number}_{flight_date}_free",
                    help="Click to select this seat."
                ):
                    selected = seat_number
    return selected

def booking_form(flight_date: str, seat_number: int):
    st.subheader(f"Booking Form — Date: {flight_date} • Seat: {seat_label(seat_number)}")
    with st.form(key="booking_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input("Full name *", max_chars=80)
            age = st.number_input("Age *", min_value=0, max_value=120, value=25, step=1)
            dob = st.date_input("Date of birth *", value=dt.date(2000, 1, 1), min_value=dt.date(1900,1,1))
        with c2:
            price = st.number_input("Ticket price (USD) *", min_value=0.0, max_value=100000.0, value=150.0, step=1.0, format="%.2f")
            st.write("")
            st.write("")
            st.info("Prices are user-defined for this demo. Adjust as needed.")

        submitted = st.form_submit_button("Confirm Booking")
        if submitted:
            if not name.strip():
                st.error("Please enter a name.")
                st.stop()
            success, msg = insert_booking(
                flight_date=flight_date,
                seat_number=seat_number,
                name=name.strip(),
                age=int(age),
                dob=dob.isoformat(),
                price=float(price),
            )
            if success:
                st.success(msg)
                # Clear selection after booking
                st.session_state.pop("selected_seat", None)
            else:
                st.error(msg)

def bookings_dashboard(default_date: Optional[str] = None):
    st.subheader("Bookings Dashboard")
    f1, f2, f3 = st.columns([1,1,1])
    with f1:
        date_filter = st.selectbox("Filter by date", options=["(All)"] + list_flight_dates(), index=0)
        date_filter = None if date_filter == "(All)" else date_filter
    with f2:
        name_filter = st.text_input("Search by name")
    with f3:
        st.write("")
        st.write("")
        if st.button("Export CSV"):
            df_export = fetch_bookings(search_name=name_filter or None, flight_date=date_filter or None)
            csv_path = "bookings_export.csv"
            df_export.to_csv(csv_path, index=False)
            st.success("Exported current view.")
            st.download_button("Download CSV", data=df_export.to_csv(index=False).encode("utf-8"), file_name=csv_path, mime="text/csv")

    df = fetch_bookings(search_name=name_filter or None, flight_date=date_filter or None)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Optional: allow cancel
    if not df.empty:
        st.divider()
        st.caption("Cancel a booking")
        cancel_id = st.number_input("Enter booking ID to cancel", min_value=0, step=1, value=0)
        if st.button("Cancel booking"):
            if cancel_id == 0:
                st.warning("Enter a valid booking ID shown in the table above.")
            else:
                ok, msg = cancel_booking(int(cancel_id))
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

# -----------------------------
# Streamlit App
# -----------------------------

def main():
    st.set_page_config(page_title="Airplane Booking (50 seats)", page_icon="✈️", layout="wide")

    st.title("✈️ Airplane Booking System")
    st.caption("50-seat flights • Next 3 months • SQLite-backed • Streamlit UI")

    # Ensure DB exists and is seeded
    init_db()
    # Only seed if needed (idempotent)
    seed_flights(days=90)

    # Sidebar: choose mode
    mode = st.sidebar.radio("Mode", ["Book a seat", "View bookings"], index=0)

    # Choose flight date
    all_dates = list_flight_dates()
    today_iso = dt.date.today().isoformat()
    default_idx = all_dates.index(today_iso) if today_iso in all_dates else 0

    if mode == "Book a seat":
        st.sidebar.header("Choose flight date")
        flight_date = st.sidebar.selectbox("Flight date", options=all_dates, index=default_idx)

        st.markdown(f"### Selected flight date: **{flight_date}**")

        # Seat map
        selected_seat = render_seat_map(flight_date)

        # Persist selection across reruns
        if selected_seat is not None:
            st.session_state["selected_seat"] = selected_seat

        if "selected_seat" in st.session_state:
            st.success(f"Selected seat: {seat_label(st.session_state['selected_seat'])}")
            booking_form(flight_date, st.session_state["selected_seat"])
        else:
            st.info("Select an available seat to start the booking.")

        st.divider()
        st.markdown("#### Seats summary")
        booked = booked_seats_for_date(flight_date)
        st.write(f"**Booked:** {len(booked)} / 50")
        if booked:
            st.write("Booked seats:", ", ".join(seat_label(s) for s in booked))

    else:
        bookings_dashboard(default_date=today_iso)

    st.divider()
    st.caption("Demo app — for production use, add authentication and concurrency controls.")

if __name__ == "__main__":
    main()
