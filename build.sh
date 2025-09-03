docker login -u cristianofontanadata

passar token de acesso  está no .env 

docker build -t mr_shop .

docker tag mr_shop:latest cristianofontanadata/mr_shop:latest

docker push cristianofontanadata/mr_shop:latest