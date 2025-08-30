docker login -u cristianofontanadata

passar token de acesso  está no .env 

docker build -t gamma_store .

docker tag gamma_store:latest cristianofontanadata/gamma_store:latest

docker push cristianofontanadata/gamma_store:latest