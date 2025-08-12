docker login -u cristianofontanadata

passar token de acesso  está no .env 

docker build -t upgradefone .

docker tag upgradefone:latest cristianofontanadata/upgradefone:latest

docker push cristianofontanadata/upgradefone:latest