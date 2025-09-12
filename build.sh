docker login -u cristianofontanadata

passar token de acesso  está no .env 

docker build -t iclub_castanhal .

docker tag iclub_castanhal:latest cristianofontanadata/iclub_castanhal:latest

docker push cristianofontanadata/iclub_castanhal:latest