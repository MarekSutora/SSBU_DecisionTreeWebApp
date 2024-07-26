import os
import pandas as pd
from flask import Flask, render_template, request, redirect, flash, url_for, send_from_directory
from flask_wtf import FlaskForm
from wtforms import FloatField, SelectField, SubmitField
from wtforms.validators import InputRequired
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from graphviz import Source
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

models = {
    "Logistic Regression": LogisticRegression(solver='liblinear'),
    "k-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Trees": DecisionTreeClassifier(criterion='entropy', max_depth=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Neural Network": MLPClassifier(max_iter=1000),
}

dataset_name = None
feature_importances = {}
scores = {}
g_model = None
dataset = None


@app.route("/", methods=["GET", "POST"])
def index():
    global dataset
    print("index")
    if dataset is not None:  # dataset has been loaded into the application
        form = create_form(dataset.columns[:-1])
        if form.validate_on_submit():

            input_data = [form.data[col] for col in dataset.columns[:-1]]
            model_name = form.data['model_name']

            prediction = models[model_name].predict([input_data])[0]

            tree_img = generate_tree_image() if model_name == "Decision Trees" else None

            return render_template("index.html", form=form, scores=scores, prediction=prediction,
                                   tree_image=tree_img, feature_importances=feature_importances,
                                   dataset_name=dataset_name, outcome_label=dataset.columns[-1])
        return render_template("index.html", form=form, scores=scores, feature_importances=feature_importances,
                               dataset_name=dataset_name, outcome_label=dataset.columns[-1])
    else:
        return redirect(url_for('upload_file'))


@app.route("/upload", methods=['GET', 'POST'])
def upload_file():
    global dataset_name
    print("upload_file")
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            dataset_name = os.path.splitext(filename)[0]  # splits the filename and the extension
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            process_file(file_path)
            return redirect(url_for('index'))
    return render_template('upload.html')


def process_file(file_path):
    global dataset, g_model
    dataset = pd.read_csv(file_path)

    dataset = dataset.dropna()

    x = dataset.iloc[:, :-1]  # select all rows and all columns except the last one
    y = dataset.iloc[:, -1]  # selects the last column of the DataFrame

    global scores
    scores = {}
    for name, model in models.items():
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        model.fit(x_train, y_train)
        cv_scores = cross_val_score(model, x_train, y_train, cv=10, scoring='accuracy')
        percentage = round(cv_scores.mean() * 100, 2)
        scores[name] = percentage

        if name == "Decision Trees":
            g_model = model
    calculate_feature_importances()

    scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}
    # sorts the scores dictionary by value in descending order


def create_form(columns):
    class DynamicForm(FlaskForm):
        pass
    for column in columns:
        clean_column = column.replace(" ", "_")  # Replace spaces with underscores
        setattr(DynamicForm, clean_column, FloatField(column, validators=[InputRequired()]))
    setattr(DynamicForm, 'model_name', SelectField('Model', choices=[(k, k) for k in models.keys()]))
    setattr(DynamicForm, 'submit', SubmitField('Predict'))
    return DynamicForm()


@app.route("/tree_image/<filename>")
def tree_image(filename):
    return send_from_directory("static", filename)


def generate_tree_image():
    graph_data = tree.export_graphviz(g_model, out_file=None, feature_names=dataset.columns[:-1],
                                      class_names=["Negative", "Positive"], filled=True)
    graph = Source(graph_data, format="svg")

    tree_filename = "decision_tree"
    graph.render(os.path.join("static", tree_filename), format="svg", view=False, cleanup=True)

    return tree_filename + ".svg"


def calculate_feature_importances():
    for name, model in models.items():
        if name in ["Decision Trees", "Random Forest"]:
            importance = model.feature_importances_
            print("importance")
            indices = np.argsort(importance)[::-1]
            print(indices)
            top_features = [dataset.columns[i] for i in indices[:3]]

            feature_importances[name] = top_features


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


if __name__ == "__main__":
    app.run(debug=True)
