docker login -u cristianofontanadata

passar token de acesso  está no .env 

docker build -t five_store .

docker tag five_store:latest cristianofontanadata/five_store:latest

docker push cristianofontanadata/five_store:latest