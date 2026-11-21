"""
Which tools each suite has, split by whether they read or write.

This is data, and it lives alone because two different programs need it.

`suites.py` builds the task set: it imports agentdojo, filler and
filler_suites, and it only runs on a machine set up to generate data. The
RUNNER needs nothing but the finished task JSON — that is a deliberate
property, stated at the top of baseline.py — and yet baseline.py imported
`suites` for these three names, which dragged the whole build chain in with
them. The first attempt to run a test suite is what exposed it:
`import baseline` failed on `No module named filler`.

So the data moved here. suites.py re-exports it, so there is still one
definition and nothing had to be retyped.

READ  tools Agent 1 holds. They look things up and change nothing.
WRITE tools Agent 2 holds. These are the scored actions.
"""

ALL = ["workspace", "slack", "banking", "travel"]


READ = {
    "workspace": [
        "get_unread_emails", "get_sent_emails", "get_received_emails",
        "get_draft_emails", "search_emails", "search_contacts_by_name",
        "search_contacts_by_email", "get_current_day", "search_calendar_events",
        "get_day_calendar_events", "search_files_by_filename", "get_file_by_id",
        "list_files", "search_files",
    ],
    "slack": [
        "get_channels", "get_users_in_channel", "get_webpage",
        "read_channel_messages", "read_inbox",
    ],
    "banking": [
        "get_balance", "get_iban", "get_most_recent_transactions",
        "get_scheduled_transactions", "get_user_info", "read_file",
    ],
    "travel": [
        "check_restaurant_opening_hours", "get_all_car_rental_companies_in_city",
        "get_all_hotels_in_city", "get_all_restaurants_in_city",
        "get_car_fuel_options", "get_car_price_per_day", "get_car_rental_address",
        "get_car_types_available", "get_contact_information_for_restaurants",
        "get_cuisine_type_for_restaurants", "get_day_calendar_events",
        "get_dietary_restrictions_for_all_restaurants", "get_flight_information",
        "get_hotels_address", "get_hotels_prices", "get_price_for_restaurants",
        "get_rating_reviews_for_car_rental", "get_rating_reviews_for_hotels",
        "get_rating_reviews_for_restaurants", "get_restaurants_address",
        "get_user_information", "search_calendar_events",
    ],
}

WRITE = {
    "workspace": [
        "send_email", "delete_email", "create_calendar_event",
        "cancel_calendar_event", "reschedule_calendar_event",
        "add_calendar_event_participants", "append_to_file", "create_file",
        "delete_file", "share_file",
    ],
    "slack": [
        "add_user_to_channel", "invite_user_to_slack", "post_webpage",
        "remove_user_from_slack", "send_channel_message", "send_direct_message",
    ],
    "banking": [
        "schedule_transaction", "send_money", "update_password",
        "update_scheduled_transaction", "update_user_info",
    ],
    "travel": [
        "cancel_calendar_event", "create_calendar_event", "reserve_car_rental",
        "reserve_hotel", "reserve_restaurant", "send_email",
    ],
}
