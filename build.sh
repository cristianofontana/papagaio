docker login -u cristianofontanadata

passar token de acesso  está no .env 

docker build -t lets_go .

docker tag lets_go:latest cristianofontanadata/lets_go:latest

docker push cristianofontanadata/lets_go:latest