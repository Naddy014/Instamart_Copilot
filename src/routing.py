PRIORITY_CATEGORIES = {"fraud"}


def route_ticket(category, segment):
    if category == "fraud":
        return "Security Team", "High"
    if segment == "Premium":
        return "Priority Support", "High"
    return {
        "billing": ("Billing Team", "Normal"),
        "delivery": ("Delivery Operations", "Normal"),
        "technical": ("Technical Support", "Normal"),
    }.get(category, ("General Support", "Normal"))
