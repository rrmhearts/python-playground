from flask import Flask, render_template, request, redirect, url_for
from forms import Todo
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SECRET_KEY'] = 'password'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///tmp/test.db'
db = SQLAlchemy(app)

class TodoModel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(80), nullable=False)

    def __repr__(self):
        return '<Todo %r>' % self.content

@app.route('/', methods=['GET', 'POST'])
def index():
    todos = TodoModel.query.all()
    if request.method == 'POST':
        data = request.form
        firstname = data['firstname']
        lastname = data['lastname']
        print('Received', data)
        return redirect(url_for('name', firstname=firstname))

    return render_template('home.j2', \
        request_method=request.method, \
        todos=todos)

@app.route('/name/<string:firstname>')
def name(firstname):
    return f'{firstname}'

@app.route('/todo', methods=['GET', 'POST'])
def todo():
    todo_form = Todo()
    if todo_form.validate_on_submit():
        todo = TodoModel(content=todo_form.content.data)
        db.session.add(todo)
        db.session.commit()
        return redirect('/')

    return render_template('todo.j2', form=todo_form)

@app.route('/about', methods=['GET', 'POST'])
def about():
    return render_template('about.j2')

@app.route('/<string:named>')
def greet(named):
    return f'Hello {named}'

if __name__ == '__main__':
    app.run(debug=True)