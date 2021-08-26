from flask import Flask, render_template, request, redirect, url_for
from forms import Todo

app = Flask(__name__)
app.config['SECRET_KEY'] = 'password'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        data = request.form
        firstname = data['firstname']
        lastname = data['lastname']
        print('Received', data)
        return redirect(url_for('name', firstname=firstname))

    return render_template('home.j2', \
        request_method=request.method, \
        list_of_names=['Elon', 'Jeff', 'Ash'])

@app.route('/name/<string:firstname>')
def name(firstname):
    return f'{firstname}'

@app.route('/todo', methods=['GET', 'POST'])
def todo():
    todo_form = Todo()
    if todo_form.validate_on_submit():
        print(todo_form.content.data)
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