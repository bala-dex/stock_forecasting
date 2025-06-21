
import panel as pn
import pandas as pd
import joblib

pn.extension()

# Load model
model = joblib.load("lgbm_demand_model.pkl")

# UI widgets
bar = pn.widgets.Select(name='Bar Name', options=['Bar A', 'Bar B'])
brand = pn.widgets.Select(name='Brand Name', options=['Brand X', 'Brand Y'])
day = pn.widgets.IntSlider(name='Day of Week', start=0, end=6)
hour = pn.widgets.IntSlider(name='Hour of Day', start=0, end=23)
lag_1 = pn.widgets.FloatInput(name='Yesterday Consumption (ml)', value=300.0)
lag_2 = pn.widgets.FloatInput(name='Day Before Yesterday (ml)', value=250.0)
roll_mean = pn.widgets.FloatInput(name='3-Day Avg Consumption (ml)', value=280.0)
roll_std = pn.widgets.FloatInput(name='3-Day Std Dev (ml)', value=15.0)

output = pn.pane.Markdown("")

def predict():
    df = pd.DataFrame([{
        'day_of_week': day.value,
        'hour': hour.value,
        'is_weekend': 1 if day.value in [5,6] else 0,
        'lag_1': lag_1.value,
        'lag_2': lag_2.value,
        'rolling_mean_3': roll_mean.value,
        'rolling_std_3': roll_std.value,
        'Bar Name_Bar B': 1 if bar.value == 'Bar B' else 0,
        'Brand Name_Brand Y': 1 if brand.value == 'Brand Y' else 0
    }])
    pred = model.predict(df)[0]
    output.object = f"### 📦 Predicted Consumption: **{pred:.2f} ml**"

predict_btn = pn.widgets.Button(name='Predict', button_type='primary')
predict_btn.on_click(lambda event: predict())

pn.Column(
    "# 🍸 Bar Inventory Demand Predictor",
    bar, brand, day, hour, lag_1, lag_2, roll_mean, roll_std,
    predict_btn, output
).servable()
