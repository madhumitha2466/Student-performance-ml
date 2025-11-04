import json
import numpy as np
import joblib
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

# ---------------------- Flask setup ----------------------
app = Flask(__name__)
app.secret_key = 'your_secret_key'
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# ---------------------- Config load ----------------------
with open('config.json') as f:
    config = json.load(f)

all_features = config["features"]
ranges = config["ranges"]

# ---------------------- Login setup ----------------------
users_db = {
    'student': {'password': '123', 'role': 'student'},
    'teacher': {'password': '123', 'role': 'teacher'},
    'admin':   {'password': '123', 'role': 'admin'}
}
role_features = {role: all_features for role in users_db}

class User(UserMixin):
    def __init__(self, username, role):
        self.id = username
        self.role = role

@login_manager.user_loader
def load_user(user_id):
    user_info = users_db.get(user_id)
    return User(user_id, user_info['role']) if user_info else None

# ---------------------- Routes ----------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user_info = users_db.get(username)
        if user_info and user_info['password'] == password:
            login_user(User(username, user_info['role']))
            return redirect(url_for('index'))
        flash('Invalid login credentials.')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# ---------------------- Models load ----------------------
clf = joblib.load("static/dropout_model.pkl")   # Logistic Regression
reg = joblib.load("static/grade_model.pkl")     # Random Forest

# ---------------------- Helpers ----------------------
def grade_letter(x):
    if x >= 9: return "A+"
    elif x >= 8: return "A"
    elif x >= 7: return "B"
    elif x >= 6: return "C"
    elif x >= 5: return "D"
    else: return "F"

@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    features = role_features.get(current_user.role, all_features)

    if request.method == "POST":
        vals = []
        errors = []

        # Validate all inputs and show all errors together
        for f in features:
            value = request.form.get(f)
            try:
                val = float(value)
                mn, mx = ranges[f]
                if not (mn <= val <= mx):
                    errors.append(f"{f.replace('_',' ').title()} must be between {mn} and {mx}.")
                else:
                    vals.append(val)
            except:
                errors.append(f"Invalid input for {f.replace('_',' ').title()}. Enter a valid number.")

        if errors:
            for e in errors:
                flash(e)
            return redirect(url_for('index'))

        # Prediction
        arr = np.array(vals).reshape(1, -1)
        try:
            dropout_prob = clf.predict_proba(arr)[0][1]  # Logistic Regression output
            grade_pred = reg.predict(arr)[0]
            importances = np.abs(clf.coef_).flatten().tolist()  # Feature importances (LR weights)
            grade = grade_letter(grade_pred)
        except Exception as e:
            flash(f"Prediction error: {str(e)}")
            return redirect(url_for('index'))

        # Store results
        session['result_data'] = {
            'dropout': round(dropout_prob * 100, 2),
            'grade': grade,
            'score': round(grade_pred, 2),
            'importances': importances,
            'fields': features
        }
        return redirect(url_for('results'))

    return render_template("index.html", features=features)

@app.route("/results")
@login_required
def results():
    if 'result_data' not in session:
        return redirect(url_for('index'))
    data = session.pop('result_data')
    return render_template("result.html", **data, zip=zip)

# ---------------------- Main ----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

