from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def landing_page():
    return render_template('cavhire_landing_page.html')

@app.route('/opportunities_map')
def opportunities_map():
    return render_template('uva_map.html')

@app.route('/dynamic_calendar')
def dynamic_calender():
    return render_template('dynamic_calendar.html')

@app.route('/analytics')
def analytics():
    return render_template('visualizations.html')

@app.route('/alternate-map')
def dark_map():
    return render_template('bt_2.html')
