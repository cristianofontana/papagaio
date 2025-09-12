docker login -u cristianofontanadata

passar token de acesso  está no .env 

docker build -t iclub_belem .

docker tag iclub_belem:latest cristianofontanadata/iclub_belem:latest

docker push cristianofontanadata/iclub_belem:latest