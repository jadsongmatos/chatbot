# ChatBOT

Trata-se de uma aplicação web Flask que se integra ao Dialogflow do Google Cloud e utiliza o modelo Bloom do projeto BigScience para geração de texto. O aplicativo cria novas intenções no Dialogflow com base nas consultas do usuário e usa o texto gerado como resposta.

Aqui está uma breve explicação dos principais componentes:

1. Importe as bibliotecas necessárias e crie uma instância do aplicativo Flask.
2. Carregue o modelo Bloom e o tokenizer.
3. Defina o ID do projeto do Google Cloud e carregue as credenciais.
4. Crie um cliente de intenções do Dialogflow.
5. Defina a função `generate(prompt)` que recebe um prompt e gera uma resposta usando o modelo Bloom.
6. Defina a função `create_intent(pergunta)` que recebe uma pergunta e cria uma nova intenção no Dialogflow com a resposta gerada como o texto da mensagem.
7. Defina a função `webhook()`, que é chamada quando o aplicativo recebe uma solicitação POST. Essa função extrai a consulta do usuário, cria uma nova intenção usando a função `create_intent()` e retorna uma resposta JSON.
8. Execute o aplicativo Flask.

install dependencies:

```bash
pip install -r requirements.txt
```

run the app:

```bash
python app.py
```

```bash
uwsgi --ini uwsgi.ini --socket 0.0.0.0:5000 --protocol=http
```