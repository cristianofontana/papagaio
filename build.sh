docker login -u cristianofontanadata

passar token de acesso  está no .env 

docker build -t mobifix .

docker tag mobifix:latest cristianofontanadata/mobifix:latest

docker push cristianofontanadata/mobifix:latest