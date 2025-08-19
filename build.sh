docker login -u cristianofontanadata

passar token de acesso  está no .env 

docker build -t weber_store .

docker tag weber_store:latest cristianofontanadata/weber_store:latest

docker push cristianofontanadata/weber_store:latest