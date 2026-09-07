"""Generate a sample Excel file in the exact production column format.

Replace data/defects.xlsx with the real export whenever it is ready - nothing
else in the pipeline changes.
"""
import random
from datetime import date, timedelta

import pandas as pd

import config

DESCRIPTIONS = [
    "Customer reports tyre damage on front left. Sidewall cut visible after pothole hit.",
    "Tyre me se awaaz aa rahi hai while driving above 60 kmph. Customer not satisfied.",
    "Rear right wheel rim bent. Alloy wheel damaged, vibration felt in steering.",
    "Tire pressure dropping frequently. Slow puncture suspected in rear left.",
    "Tyer worn out unevenly on front axle. Alignment issue suspected.",
    "Wheel balancing weight missing. Vibration at high speed reported by customer.",
    "Tread depth below limit on both front tyres at 12000 km only. Premature wear.",
    "Gaadi ka tyre burst ho gaya highway pe. Customer claims manufacturing defect.",
    "Puncture repeated 3 times in same tyre within one month. Rim seating suspected.",
    "Alloy wheel corrosion observed near valve area. Paint peeling from rim surface.",
    "Brake pad worn out prematurely at 15000 km. Grinding noise while braking.",
    "Brake me se awaaz aati hai jab bhi brake lagate hain. Squealing sound.",
    "ABS warning light coming on dashboard intermittently. Customer complaint.",
    "Brake fluid leakage observed near rear left caliper during routine service.",
    "AC not cooling properly. Compressor cycling on and off frequently.",
    "AC se thandi hawa nahi aa rahi. Gas leakage suspected in condenser.",
    "Blower motor making rattling noise at high speed setting. AC vent airflow low.",
    "Paint peeling near driver side door handle. Clear coat lifting observed.",
    "Body panel gap uneven on rear door. Paint mismatch visible in sunlight.",
    "Rust spots on bonnet edge within 8 months of delivery. Transit damage suspected.",
    "Scratch marks on rear bumper noticed at PDI. Handling damage at yard.",
    "Engine oil leakage from sump gasket. Oil stains on customer parking observed.",
    "Engine check light illuminated. Customer reports rough idling and vibration.",
    "Engine start nahi ho raha subah me. Cold start problem reported repeatedly.",
    "Excessive smoke from exhaust during acceleration. Turbo issue suspected.",
    "Battery drain overnight. Customer needs jump start every second day.",
    "Headlamp condensation observed inside cluster. Moisture ingress from seal.",
    "Power window switch not working on rear left door. Electrical fault.",
    "Infotainment screen goes blank randomly. Software glitch reported by customer.",
    "Horn not working intermittently. Wiring harness connection loose.",
    "Suspension noise over speed breakers. Knocking sound from front left strut.",
    "Shocker se awaaz aa rahi hai. Suspension bush worn out at 20000 km.",
    "Steering pulling to left side while driving straight. Alignment done twice.",
    "Clutch pedal hard to press. Customer reports difficulty in gear shifting.",
    "Gear shifting problem in 2nd gear. Grinding noise while downshifting.",
    "Seat fabric torn near stitching on rear bench. Quality issue reported.",
    "Door trim rattling noise on rough roads. Clip loose from factory.",
    "Wiper blade smearing on windshield. Rubber deteriorated within 6 months.",
    "Fuel efficiency lower than claimed. Customer dissatisfied with mileage.",
    "Coolant level dropping without visible leak. Radiator inspection required.",
    # multi-defect paragraphs - these exercise per-sentence chunking
    "Noise from front wheel while braking. Also AC not cooling properly. Paint peeling near driver door handle.",
    "Customer states tyre me se awaaz aa rahi hai. Rear left wheel rim bent, alloy damaged during pothole hit. Suspension bush also worn out.",
    "Engine oil leak observed at sump. Brake pads worn out. Battery weak, needs replacement.",
    "PDI audit: scratch on rear bumper, tyre pressure low in all four wheels, wiper blade defective.",
    "Multiple issues reported. Tire tread wear uneven. Steering vibration at 80 kmph. Wheel alignment out of spec.",
]

SOURCES = ["Dealer Verbatim", "Internal Audit", "Warranty Claim", "Field Report", "PDI Check"]
ATTRIBUTIONS = ["Manufacturing", "Supplier", "Transit Damage", "Dealer Handling", "Customer Misuse", "Under Investigation"]
SHOPS = ["Pune-Kothrud", "Mumbai-Andheri", "Delhi-Rohini", "Chennai-Guindy", "Jaipur-Vaishali", "Kolkata-Salt Lake"]
AUDITORS = ["R. Sharma", "A. Iyer", "M. Khan", "S. Patil", "N. Verma", "P. Desai"]


def main(n: int = 120) -> None:
    rng = random.Random(42)
    start = date(2026, 1, 1)
    records = []
    for i in range(n):
        records.append({
            "Source": rng.choice(SOURCES),
            "Problem Description": DESCRIPTIONS[i % len(DESCRIPTIONS)],
            "Attribution": rng.choice(ATTRIBUTIONS),
            "Shop": rng.choice(SHOPS),
            "Auditor": rng.choice(AUDITORS),
            "Reported Date": (start + timedelta(days=rng.randint(0, 240))).isoformat(),
        })
    df = pd.DataFrame(records, columns=config.COLUMNS)
    df.to_excel(config.DATA_FILE, index=False)
    print(f"wrote {len(df)} rows -> {config.DATA_FILE}")


if __name__ == "__main__":
    main()
