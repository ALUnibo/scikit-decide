import numpy as np
import torch as th
from torch import nn
from skdecide import Value
from skdecide.hub.domain.flight_planning.domain import D
from pygeodesy.ellipsoidalVincenty import LatLon
from openap.extra.aero import distance, ft


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x


def train(model, dataloader, loss_fn, optimizer):
    model.train()  # Set the model to training mode
    for batch in dataloader:
        # Assume that dataloader returns a tuple of (inputs, targets)
        inputs, targets = batch
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def heuristic(domain, s: D.T_state, heuristic_name: str = None) -> Value[D.T_value]:
    pos = s.trajectory.iloc[-1]

    lat_to, lon_to, alt_to = domain.lat2, domain.lon2, domain.alt2
    lat_start, lon_start, alt_start = domain.lat1, domain.lon1, domain.alt1

    # Compute distance in meters
    distance_to_goal = LatLon.distanceTo(
        LatLon(pos["lat"], pos["lon"], height=pos["alt"] * ft),  # alt ft -> meters
        LatLon(lat_to, lon_to, height=alt_to * ft),  # alt ft -> meters
    )
    distance_to_start = LatLon.distanceTo(
        LatLon(pos["lat"], pos["lon"], height=pos["alt"] * ft),  # alt ft -> meters
        LatLon(lat_start, lon_start, height=alt_start * ft),  # alt ft -> meters
    )

    return Value(cost=0)


def flying(
    self, from_: pd.DataFrame, to_: Tuple[float, float, int]
) -> pd.DataFrame:
    """Compute the trajectory of a flying object from a given point to a given point

    # Parameters
        from_ (pd.DataFrame): the trajectory of the object so far
        to_ (Tuple[float, float]): the destination of the object

    # Returns
        pd.DataFrame: the final trajectory of the object
    """
    pos = from_.to_dict("records")[0]

    lat_to, lon_to, alt_to = to_[0], to_[1], to_[2]
    dist_ = distance(
        pos["lat"], pos["lon"], lat_to, lon_to, h=(alt_to - pos["alt"]) * ft
    )

    data = []
    epsilon = 100
    dt = 600
    dist = dist_
    loop = 0

    while dist > epsilon:
        # bearing of the plane
        bearing_degrees = aero_bearing(pos["lat"], pos["lon"], lat_to, lon_to)

        # wind computations & A/C speed modification
        we, wn = 0, 0
        temp = 273.15
        if self.weather_interpolator:
            time = pos["ts"] % (3_600 * 24)

            # wind computations
            wind_ms = self.weather_interpolator.interpol_wind_classic(
                lat=pos["lat"], longi=pos["lon"], alt=alt_to, t=time
            )
            we, wn = wind_ms[2][0], wind_ms[2][1]

            # temperature computations
            temp = self.weather_interpolator.interpol_field(
                [pos["ts"], pos["alt"], pos["lat"], pos["lon"]], field="T"
            )

        wspd = sqrt(wn * wn + we * we)

        tas = mach2tas(self.mach, alt_to * ft)  # alt ft -> meters

        gs = compute_gspeed(
            tas=tas,
            true_course=radians(bearing_degrees),
            wind_speed=wspd,
            wind_direction=3 * math.pi / 2 - atan2(wn, we),
        )

        if gs * dt > dist:
            # Last step. make sure we go to destination.
            dt = dist / gs
            ll = lat_to, lon_to
        else:
            ll = latlon(
                pos["lat"],
                pos["lon"],
                d=gs * dt,
                brg=bearing_degrees,
                h=alt_to * ft,
            )

        values_current = {
            "mass": pos["mass"],
            "alt": pos["alt"],
            "speed": tas / kts,
            "temp": temp,
        }

        pos["fuel"] = self.perf_model.compute_fuel_consumption(
            values_current,
            delta_time=dt,
            path_angle=math.degrees((alt_to - pos["alt"]) * ft / (gs * dt)),
            # approximation for small angles: tan(alpha) ~ alpha
        )

        mass = pos["mass"] - pos["fuel"]

        # get new weather interpolators
        if pos["ts"] + dt >= (3_600.0 * 24.0):
            if self.weather_date:
                if self.weather_date == self.initial_date:
                    self.weather_date = self.weather_date.next_day()
                    self.weather_interpolator = self.get_weather_interpolator()
        else:
            if self.weather_date != self.initial_date:
                self.weather_date = self.weather_date.previous_day()
                self.weather_interpolator = self.get_weather_interpolator()

        new_row = {
            "ts": (pos["ts"] + dt),
            "lat": ll[0],
            "lon": ll[1],
            "mass": mass,
            "mach": self.mach,
            "fuel": pos["fuel"],
            "alt": alt_to,  # to be modified
        }

        dist = distance(
            ll[0],
            ll[1],
            lat_to,
            lon_to,
            h=(pos["alt"] - alt_to) * ft,  # height difference in m
        )

        if dist < dist_:
            data.append(new_row)
            dist_ = dist
            pos = data[-1]

        else:
            dt = int(dt / 10)
            print("going in the wrong part.")
            assert dt > 0

        loop += 1

    return pd.DataFrame(data)

if __name__ == "__main__":
    LatLon.distanceTo(LatLon(0, 0, height=0), LatLon(1, 1, height=0))

