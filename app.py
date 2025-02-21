from apiflask import APIFlask

app = APIFlask(__name__)


@app.route("/")
def hello_world():
    return {"message": "Hello World!"}
