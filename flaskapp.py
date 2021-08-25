from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('hello.j2', list_of_names=['Elon', 'Jeff', 'Ash'])

@app.route('/<string:name>')
def greet(name):
    return f'Hello {name}'

if __name__ == '__main__':
    app.run(debug=True)