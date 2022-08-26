# 1) Importação de Bibliotecas

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

# 2) Importação de Datasets

## 2.1) Dataset `Propostas`

df1 = pd.read_csv('Dados_brutos\dexproposals_20220405.csv', header=None,\
 names=['id','admin', 'name','pax', 'start_date', 'end_date','start_time','end_time','address','current_version','is_active', 'expires_at','created_at','updated_at','deleted_at'])
df2 = pd.read_csv('Dados_brutos\dexproposals_20220413.csv', header=None,\
 names=['id','admin', 'name','pax', 'start_date', 'end_date','start_time','end_time','address','current_version','is_active', 'expires_at','created_at','updated_at','deleted_at'])
df3 = pd.read_csv('Dados_brutos\dexproposals-20220518.csv', header=None,\
 names=['id','admin', 'name','pax', 'start_date', 'end_date','start_time','end_time','address','current_version','is_active', 'expires_at','created_at','updated_at','deleted_at'])
df_propostas = pd.concat([df1,df2,df3])

## 2.2) Dataset `Itens de propostas`

df1 = pd.read_csv('Dados_brutos\dexproposalitems_20220405.csv', header=None,\
 names=['id','client_proposal_id','original_item_id','version','name','sale_price','supplier_sale_price', 'buyer_price',\
       'duration','quantity','price_method', 'editable_quantity', 'min_quantity','max_quantity', 'additional_fee_name', 'additional_fee',\
       'supplier_name', 'obs', 'show_images', 'section_id','section_name','section_position', 'complexity', 'discount_percent','final_buyer_price', 'json_object'])
df2 = pd.read_csv('Dados_brutos\dexproposalitems_20220413.csv', header=None,\
 names=['id','client_proposal_id','original_item_id','version','name','sale_price','supplier_sale_price', 'buyer_price',\
       'duration','quantity','price_method', 'editable_quantity', 'min_quantity','max_quantity', 'additional_fee_name', 'additional_fee',\
       'supplier_name', 'obs', 'show_images', 'section_id','section_name','section_position', 'complexity', 'discount_percent','final_buyer_price', 'json_object'])
df3 = pd.read_csv('Dados_brutos\dexproposalitems-20220518.csv', header=None,\
 names=['id','client_proposal_id','original_item_id','version','name','sale_price','supplier_sale_price', 'buyer_price',\
       'duration','quantity','price_method', 'editable_quantity', 'min_quantity','max_quantity', 'additional_fee_name', 'additional_fee',\
       'supplier_name', 'obs', 'show_images', 'section_id','section_name','section_position', 'complexity', 'discount_percent','final_buyer_price', 'json_object'])
df_itens = pd.concat([df1,df2,df3])

## 2.3) Dataset dos `pedidos`

df1 = pd.read_csv('Dados_brutos\dexorders_20220405.csv',header=None,\
  names=['order_id','sale_code','client_proposal_id','name_(nome da proposta)','obs','active','total_buyer_price','created_at','updated_at','client_proposal_id.1',\
         'version','name_(nome do item)','description', 'buyer_price','duration','quantity','price_method','editable_quantity','min_quantity','max_quantity',\
         'supplier_sale_price','rate_price','sale_price','final_supplier_sale_price','final_rate_price','final_sale_price','additional_fee_name','additional_fee',\
         'supplier_name','company_name','supplier_email','obs.1','show_images','final_buyer_price'])
df2 = pd.read_csv('Dados_brutos\dexorders_20220413.csv',header=None,\
  names=['order_id','sale_code','client_proposal_id','name_(nome da proposta)','obs','active','total_buyer_price','created_at','updated_at','client_proposal_id.1',\
         'version','name_(nome do item)','description', 'buyer_price','duration','quantity','price_method','editable_quantity','min_quantity','max_quantity',\
         'supplier_sale_price','rate_price','sale_price','final_supplier_sale_price','final_rate_price','final_sale_price','additional_fee_name','additional_fee',\
         'supplier_name','company_name','supplier_email','obs.1','show_images','final_buyer_price'])
df3 = pd.read_csv('Dados_brutos\dexorders-20220518.csv',header=None,\
  names=['order_id','sale_code','client_proposal_id','name_(nome da proposta)','obs',\
         'active','total_buyer_price','created_at','updated_at','client_proposal_id.1',\
         'version','name_(nome do item)','description', 'buyer_price','duration','quantity',\
         'price_method','editable_quantity','min_quantity','max_quantity',\
         'supplier_sale_price','rate_price','sale_price','final_supplier_sale_price',\
         'final_rate_price','final_sale_price','additional_fee_name','additional_fee',\
         'supplier_name','company_name','supplier_email','obs.1','show_images','final_buyer_price'])
df_orders = pd.concat([df1,df2,df3])

# 3) Exploração e limpeza de dados

# função para trocar '\N' por valores nulos
def Troc_Null(df):
  for col in df.columns:
    df.loc[ df[col] == r'\N', col ] = np.nan
  return df

# Transformando coluna em inteiros
def Troc_Int(df, lista):
  for col in lista:
    df[col] = df[col].astype(np.int64)
  return df

# Transformando coluna em float
def Troc_Float(df, lista):
  for col in lista:
    df[col] = df[col].astype(np.float64)
  return df

# Convertendo coluna pata formato datetime64
def Troc_Data(df, lista):
  for col in lista:
    df[col] = pd.to_datetime(df[col], format='%Y-%m-%d', errors='coerce')
  return df

## 3.1) Planilha propostas

# trocando os valores \N por NaN
df_propostas = Troc_Null(df_propostas)

###3.1.1) Coluna `ID`

# Convertendo para inteiro
df_propostas = Troc_Int(df_propostas, ['id'])

###3.1.2) Coluna `admin`

# Preenchendo valores nulos com 0 e transformando coluna em inter
df_propostas.admin = df_propostas.admin.fillna(0).astype(np.int64)

###3.1.3) Coluna `pax`

df_propostas.pax.fillna(0).astype(np.int64)

###3.1.4) Colunas `start_date`, `expires_at`,`created_at`,`updated_at`

# Convertendo colunas pata formato datetime64
df_propostas = Troc_Data(df_propostas, ['start_date','expires_at','created_at','updated_at'])

###3.1.6) Colunas `end_date`, `start_time`, `end_time`, `deleted_at`

# Deletando colunas nulas
df_propostas.drop(['end_date', 'start_time', 'end_time', 'deleted_at'], axis=1, inplace=True)

###3.1.7) Coluna `current_version`

df_propostas = Troc_Int(df_propostas, ['current_version'])

###3.1.9) Coluna `is_active`

# Convertendo valores boleanos da coluna `is_active` para 0 e 1
df_propostas.is_active = df_propostas.is_active.map({'yes':1, 'no':0})

## 3.2) Planilha itens

# trocando os valores \N por NaN
for col in df_itens.columns:
  df_itens.loc[ df_itens[col] == r'\N', col ] = np.NaN

### 3.2.1) Coluna `id`

# Limpando as linhas que não são números
df_itens = df_itens[df_itens.id.astype(str).str.isnumeric()]
df_itens = Troc_Int(df_itens, ['id'])

### 3.2.2) Colunas `discount_percent`, `json_object`

# Colunas com valores nulos para serem dropadas
df_itens.drop(['discount_percent','json_object'], axis = 1, inplace = True)

### 3.2.3) Coluna `client_proposal_id`

df_itens = Troc_Int(df_itens, ['client_proposal_id'])

### 3.2.4) Coluna `original_item_id`

df_itens = Troc_Int(df_itens, ['original_item_id'])

### 3.2.5) Coluna `version`

df_itens = Troc_Int(df_itens, ['version'])

### 3.2.6) Coluna `sale_price`

# Convertendo os valores para float
df_itens = Troc_Float(df_itens, ['sale_price'])

### 3.2.7) Coluna `supplier_sale_price`

# Convertendo os valores para float
df_itens = Troc_Float(df_itens, ['supplier_sale_price'])

df_itens.supplier_sale_price = df_itens.supplier_sale_price.fillna(0)

### 3.2.8) Coluna `buyer_price`

# Convertendo os valores para float
df_itens = Troc_Float(df_itens, ['buyer_price'])

df_itens.buyer_price = df_itens.buyer_price.fillna(0)

### 3.2.9) Coluna `duration`

# Ajustando coluna duração para horas
df_itens.duration = df_itens.duration.apply(lambda x: str(x).replace('30m', '0.5'))
df_itens.duration = df_itens.duration.apply(lambda x: str(x).replace('h', ''))
df_itens = Troc_Float(df_itens, ['duration'])

### 3.2.10) Coluna `quantity`

# Convertendo para float
df_itens = Troc_Float(df_itens, ['quantity'])

### 3.2.11) Coluna `price_method`

# Como temos o mesmo item diferenciado por maiúscula ou minúscula, vamos converter tudo para minúscula
df_itens.price_method = df_itens.price_method.apply(lambda x: str(x).lower())

# Trocando o valor x por nan
df_itens.price_method = df_itens.price_method.apply(lambda x: str(x).replace('x', 'nan'))

# trocando os valores 'nan' por NaN
df_itens.loc[ df_itens['price_method'] == r'nan', 'price_method' ] = np.NaN

### 3.2.12) Coluna `editable_quantity`

# Substituindo sim por 1 e não por 0
df_itens.editable_quantity = df_itens.editable_quantity.map({'yes':1, 'no':0})

### 3.2.13) Colunas `min_quantity` e `max_quantity`

# Valores nulos substituidos por 0
df_itens.min_quantity = df_itens.min_quantity.fillna(0)
df_itens.max_quantity = df_itens.max_quantity.fillna(0)

df_itens = Troc_Int(df_itens,['min_quantity', 'max_quantity'])

### 3.2.14) Coluna `additional_fee_name`

# Como temos o mesmo item diferenciado por maiúscula ou minúscula, vamos converter tudo para minúscula
df_itens.additional_fee_name = df_itens.additional_fee_name.apply(lambda x: str(x).lower())

# trocando os valores 'nan' e ' ' por NaN
df_itens.loc[ df_itens['additional_fee_name'] == r'nan', 'additional_fee_name' ] = np.NaN
df_itens.loc[ df_itens['additional_fee_name'] == r' ', 'additional_fee_name' ] = np.NaN
df_itens.loc[ df_itens['additional_fee_name'] == r'.', 'additional_fee_name' ] = np.NaN

### 3.2.15) Coluna `additional_fee`

# Convertendo para valores Float
df_itens = Troc_Float(df_itens, ['additional_fee'])

### 3.2.16) Coluna `supplier_name`

# Como podemos ter o mesmo item diferenciado por maiúscula ou minúscula, vamos converter tudo para minúscula
df_itens.supplier_name = df_itens.supplier_name.apply(lambda x: str(x).lower().rstrip().lstrip())

# trocando os valores 'nan' por NaN
df_itens.loc[ df_itens['supplier_name'] == r'nan', 'supplier_name' ] = np.NaN

### 3.2.17) Coluna `show_images`

df_itens.drop(['show_images'], axis=1, inplace=True)

### 3.2.18) Coluna `complexity`

df_itens.complexity = df_itens.complexity.fillna('default')

### 3.2.19) Coluna `final_buyer_price`

# Convertendo para Float
df_itens = Troc_Float(df_itens, ['final_buyer_price'])

## 3.3) Planilha Pedidos

# trocando os valores \N por NaN
for col in df_orders.columns:
  df_orders.loc[ df_orders[col] == r'\N', col ] = np.NaN

### 3.3.1) Colunas Nulas e Duplicadas

# Deletando colunas nulas
df_orders.drop(['version', 'duration', 'price_method', 'show_images'], axis=1, inplace=True)

# Deletando colunas duplicada
df_orders.drop('client_proposal_id.1', axis=1, inplace=True)

### 3.3.2) Colunas `order_id` e `client_proposal_id`

# Trocando valores para inteiros
df_orders = Troc_Int(df_orders, ['order_id','client_proposal_id'])

### 3.3.3) Colunas `supplier_sale_price`, `rate_price`, `sale_price`, `final_supplier_sale_price`, `final_rate_price`, `final_sale_price` e `final_buyer_price`

df_orders.final_sale_price = df_orders.final_sale_price.fillna(0)
df_orders.final_buyer_price = df_orders.final_buyer_price.fillna(0)

### 3.3.4) Coluna `sale_code`

# Trocar nulos para 0 e converter para inteiro
df_orders.sale_code = df_orders.sale_code.fillna(0)
df_orders = Troc_Int(df_orders,['sale_code'])

### 3.3.5) Colunas `active` e `editable_quantity`

# Convertendo valores boleanos da coluna `active` e `editable_quantity` para 0 e 1
df_orders.active = df_orders.active.map({'yes':1, 'no':0})
df_orders.editable_quantity = df_orders.editable_quantity.map({'yes':1, 'no':0})

###3.3.6) Colunas `min_quantity` e `max_quantity`

# Valores nulos substituidos por 0
df_orders.min_quantity = df_orders.min_quantity.fillna(0)
df_orders.max_quantity = df_orders.max_quantity.fillna(0)

df_orders = Troc_Float(df_orders,['min_quantity', 'max_quantity'])
df_orders = Troc_Int(df_orders,['min_quantity', 'max_quantity'])

###3.3.7) Coluna `name_(nome da proposta)`

# Como podemos ter o mesmo item diferenciado por maiúscula ou minúscula, vamos converter tudo para minúscula
df_orders['name_(nome da proposta)'] = df_orders['name_(nome da proposta)'].apply(lambda x: str(x).lower().rstrip().lstrip())

###3.3.8) Coluna `obs`

# Como podemos ter o mesmo item diferenciado por maiúscula ou minúscula, vamos converter tudo para minúscula
df_orders['obs'] = df_orders['obs'].apply(lambda x: str(x).lower().rstrip().lstrip())

# trocando os valores 'nan' por NaN
df_orders.loc[ df_orders['obs'] == r'nan', 'obs' ] = np.NaN

###3.3.9) Coluna `total_buyer_price`

# Trocando para valores Float
df_orders = Troc_Float(df_orders, ['total_buyer_price'])

### 3.3.10) Colunas `created_at` e `updated_at`

# Converter para data
df_orders = Troc_Data(df_orders,['created_at', 'updated_at'])

### 3.3.11) Colunas `name_(nome do item)` e `description`

# Como podemos ter o mesmo item diferenciado por maiúscula ou minúscula, vamos converter tudo para minúscula
df_orders['name_(nome do item)'] = df_orders['name_(nome do item)'].apply(lambda x: str(x).lower().rstrip().lstrip())
df_orders['description'] = df_orders['description'].apply(lambda x: str(x).lower().rstrip().lstrip())

# trocando os valores 'nan' por NaN
df_orders.loc[ df_orders['name_(nome do item)'] == r'nan', 'name_(nome do item)' ] = np.NaN
df_orders.loc[ df_orders['description'] == r'nan', 'description' ] = np.NaN

###3.3.12) Coluna `buyer_price`

# Trocar para float
df_orders = Troc_Float(df_orders,['buyer_price'])
df_orders.buyer_price = df_orders.buyer_price.fillna(0)

###3.3.13) Coluna `quantity`

# Convertendo para inteiro e preenchendo nan com 0
df_orders.quantity = df_orders.quantity.fillna(0)
df_orders = Troc_Int(df_orders, ['quantity'])

###3.3.14) Coluna `additional_fee_name`

# Como podemos ter o mesmo item diferenciado por maiúscula ou minúscula, vamos converter tudo para minúscula
df_orders['additional_fee_name'] = df_orders['additional_fee_name'].apply(lambda x: str(x).lower().rstrip().lstrip())

# trocando os valores 'nan' por NaN
df_orders.loc[ df_orders['additional_fee_name'] == r'nan', 'additional_fee_name' ] = np.NaN

###3.3.15) Coluna `additional_fee`

df_orders.loc[df_orders.additional_fee == 'Matias', 'additional_fee'] = np.nan

# Converter para float
df_orders = Troc_Float(df_orders, ['additional_fee'])

###3.3.16) Colunas `supplier_name`, `company_name`, `supplier_email` e `obs.1`

# Como podemos ter o mesmo item diferenciado por maiúscula ou minúscula, vamos converter tudo para minúscula
df_orders['supplier_name'] = df_orders['supplier_name'].apply(lambda x: str(x).lower().rstrip().lstrip())
df_orders['company_name'] = df_orders['company_name'].apply(lambda x: str(x).lower().rstrip().lstrip())
df_orders['supplier_email'] = df_orders['supplier_email'].apply(lambda x: str(x).lower().rstrip().lstrip())
df_orders['obs.1'] = df_orders['obs.1'].apply(lambda x: str(x).lower().rstrip().lstrip())

# trocando os valores 'nan' por NaN
df_orders.loc[ df_orders['supplier_name'] == r'nan', 'supplier_name' ] = np.NaN
df_orders.loc[ df_orders['company_name'] == r'nan', 'company_name' ] = np.NaN
df_orders.loc[ df_orders['supplier_email'] == r'nan', 'supplier_email' ] = np.NaN
df_orders.loc[ df_orders['obs.1'] == r'nan', 'obs.1' ] = np.NaN

#4) Enriquecimento de dados

##4.1) Categorização dos itens

df_itens[['id','client_proposal_id','original_item_id','version','name','sale_price','supplier_sale_price','buyer_price','duration','quantity','price_method','editable_quantity']]
df_itens[['min_quantity','max_quantity','additional_fee_name','additional_fee','supplier_name','obs','section_id','section_name','section_position','complexity','final_buyer_price']]

dfi = df_itens.copy()
dfi['name'] = dfi['name'].apply(lambda x: str(x).lower().rstrip().lstrip())
dfi['supplier_name'] = dfi['supplier_name'].apply(lambda x: str(x).lower().rstrip().lstrip())

cerveja_e_chope = ['cerveja','chope','chop','chopp','choppe','heineken']
vinhos = ['vinho','espumante','chandon']
coquetel_e_happy = ['happy','coquetel']
mixologia = ['batida','bar','mixologia','mixol','shakeira','shakeria']
outras_bebidas = ['agua','água','refri','refrigerante','suco','leite','bebidas']
almoco_ou_janta = ['almoco','almoço','janta','jantar','brunch','brunc','churrasco']
cafes = ['café','cafe','coffee','cooffee','cofe','coffe','coff','break']
buffet = ['bufet','buffet','buffett','buffeett','buff']
outros_alimentos = ['gastronomica','gastronomia','gastronômica','finger','food','pipoca','acucar','açucar','acúcar','açúcar','snack','snac','bolacha','biscoito','lanche','sanduiche','sandu','pizza','rodizio','rodizío']
bolos = ['bolo','cake','chocotone','panetone']
pascoa = ['ovo','chocolate','colher','pascoa','páscoa']
doces = ['mel','pão de mel','bombom','bom bom','bon bon','bonbon','docinho','brigadeiro','beijinho','doce','doces']
outros_confeitaria = ['sobremesa','founde','fondue','fondeu','fundi','fundue','fundeu','niver','aniversario','aniversário']
pessoas_aeb = ['garcom','garcon','garçon','garçom','bartender','comim','comin','cozinheiro','cozinheira','cozinha']
outros_aeb = ['servir','prato','talher','a&b','taça','taca','taças','tacas', 'festa','festival']
camisetas = ['camisa','camiseta','shirt','camis','polo']
calcas = ['bermuda','shorts','bemuda','calça','calca','saia']
blusas = ['blusão','blusao','jaqueta','blusa','casaco','agasalho','moleton','moletom']
calcados = ['tenis','chinelo','havaiana','tênis','sandalia','sandália','meia']
acessorios = ['viseira','luva','roupao','roupão','toalha','bone','boné','touca','toca','maskara','mascara','máscara','máskara']
bolsas = ['bag','mochila','mala','sacochila','bolsa','pochete']
fone = ['headphone','fone','ouvido','phone','head phone','head fone']
item_informatica = ['mouse','teclado','smart','watch','smartwatch','alexa','informatica']
cadernos_e_moleskine = ['apostila','caderno','caderneta','bloco','notas','moleskine','molesk','moleskyne','moleskyni','molesquine','molesquini','moleskini']
item_escritorio = ['pen','esferografica','esferográfica','lapis','lápis','lapiseira','caneta','cordão','cordões','suporte','porta','regua','bola','bolinha']
outros_itens = ['bateria','oleo','óleo','pincel','pincél','pinceis','pincéis','carimbo','ioio','iôiô','ioiô','iôio','botton','bottom','boton','botom','borrifador','cadeado','arvore','árvore','pente','moldura','ring','banho','vinil','necessaire','maleta','necesseire','necess','escova','balao','balão','baloes','balões','aspirador','airfryer','bastão','selfie','protetor','umidificador','bosqueac','alcool','álcool','alcol','álcol','cubo','capa','mimo','saco','sacos','lixo','carregador','termica','térmica','tapete']
plantas = ['flor','suculenta']
brindes = ['prêmio','premio','ima','imã','copo','caneca','chaveiro','boia','squezze','sequezze','sacola','saquinho','troféu','trofeu','trofeus','troféus','trofeis','troféis','garrafa','calendário','calendario','squeeze','squeze','sequezze','sequeezze','squeezy','zqueezzy','brinde']
livros = ['kindle','book','livro','ebook']
transporte_de_pessoas = ['motorista','van','onibus','bus','ônibus','carro','logistica','logística','logistíca']
transporte_de_itens = ['frete','facilitie','entrega','item','iten','transporte','sedex','correio','correios','coleta','retirada','envio']
local_e_espaço =['lounge','louge','estacionamento','espaço','local','jardim','jardin','auditório','auditorio','sala','hall','mato']
mesas_e_balcoes = ['mesa','balcao','balcão']
cadeiras_e_assentos = ['banqueta','cama','cadeira','assento','poltrona','poof','puf','sofa','sofá','banco']
outros_moveis =['lixeira','vela','pupito','púpito','vaso','totem','toten','totel','decoracao','decoração','decoraçao','decoracão','cenografia','corrente','mobiliario','ventilador','mobiliário','móvel','móveis','movel','moveis','decorativo','totem','toten','cone','arco','tenda','carpete']
equipamento_de_som = ['som','caixa de som','mesa de som','microfone','microphone','lapela']
equipamento_de_iluminação = ['iluminacao','iluminacação','iluminação','iluminacão','ilumincaçao','difusor','farol','luz','luminária','luminaria']
equipamento_de_video = ['câmera','camera','webcam','câmera','camera']
outros_equipamentos = ['locação','locacão','locacao','locaçao','régua','regua','energia','gerador','pedestal','equipamento','caixustre','palco','interruptor','interrupitor','equipamentos']
limpeza = ['limpeza','cleaning','faxina','esterilização','esterilizacao']
seguranca = ['seguranca','segurança','segurancas','seguranças']
testes_e_exames = ['soro','sorologico','sorológico','antigeno','antigêno','antígeno','sangue','covid','exame'] 
ambulancia_e_afins = ['ambulancia','ambulância','médico','medico','enfermagem','enfermeiro','enfermeira','paramédico','paramedico','saude','saúde']
bombeiro = ['bombeiro']
ap_musical = ['piano','atração','atracao','brass','dj','show','banda','musica','music','voz','violão','violino','violao','música']
ap_outros = ['stand','teatro','danca','dança','apresentacao','apresentação','maquina','máquina']
jogos_e_dinamicas = ['fun','momento','atividade','team','building','criança','jogo','dinamica','dinâmica','atividade','basquete']
staff_e_organ = ['orientador','orientadores','orientar','organizacao','organizaçao','organizacão','organização','monitoria','monitorar','monitoramento','acompanhamento','organizacao','organização','montagem','holding','hold','montar','organizar','monitorar','controlar','controle']
recepcao_hostness = ['contador','acesso','equipe','staff','host','hostness','recepcao','recepção','recepçao','recepcão','recepcionista','recepcio']
planejamento_e_producao = ['operação','operacao','operaçao','operacão','direcao','direção','direcão','direçao','planner','planer','planejamento','producao','produção','produçao','producão','produzir','planejar','coordenar','coordenação','coordenacao','coordenaçao','coordenacão','produtor','produtora','coordenador','coordenadora','planejador','planejadora']
traducao_e_linguas = ['traducao','tradução','traduçao','traducão','simultaneo','simultâneo','correção','correcao','correçao','correcão','revisão','revisar','interprete','intérprete']
locucoes_e_narracoes = ['narracao','narração','narraçao','narracão','locucao','locuçao','locução','locucão','legenda','legendagem','dublagem','dublador','dubladora']
tecnico_som = ['sonoralização','sonoralizacao','sonoralizaçao','sonoralizacão','sonorização','gravação','gravacao','gravaçao','gravacão','sonoplastia','somnorizacao','sonorizacao','sonorizaçao','sonorizacão','audio','áudio','aúdio']
tecnico_filmagem = ['projetor','tripe','filmagem','filmadora','filmar','video','vídeo','edicao','edição','edicão','ediçao']
tecnico_streaming = ['operador','tramissão','plataforma','streaming','transmissão','transmissao','transmisao','transmisão','live','tranmissao','link']
outros_tecnicos = ['técnica','tecnica','visita','cerimônia','cerimonia','relatorios','relatórios','treinamento','operador','fotografia','fotografo','fotografa','fotógrafo','fotógrafa','foto','design','designer','engenharia','engenheiro','arquiteto','arquiteta','arquitetura','deseho','maquete','mock','prototipo','protótipo',]
banners_e_faixas = ['fita','banner','baner','faixa','roll','flag','wing','back','bandeira']
outros_divulgacao = ['acrílica','acrilica','acrilíca','acrilíco','acrilico','acrílico','painel','painél','paineis','painéis','cordao','cordoes','cordões','cordão','display','painel','divulgacao','divulgaçao','lambe','poster','posters','divulgação','banner','adesivo','acrílico','acrilico','acrilíco','display,']
outros_impressos = ['etiqueta','sticker','cartinha','cartinhas','papel','impressaões','certificado','impressão','folha','impressao','tag','impresao','impressao','colorido','P&B','PeB','ingresso','envelope','cracha','crachás','crachá','crachas','carta','impressões','impressoes']
flyer_folder = ['flyer','folder','panfleto','cartaz','folheto']
cartao_e_ingresso = ['cartao','cartão','ingresso','pulseira','convite','credencial','credenciais']
caixa_e_pacote = ['caixa','caixinha','caixote','embalagem','pacote','box']
kits_e_cestas= ['kit','cesta']

dfi['item_orcado'] = 'Outros'
for i in cerveja_e_chope:
  dfi['item_orcado']= dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'cerveja_e_chope')
for i in vinhos:
  dfi['item_orcado'] = dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'vinhos')
for i in coquetel_e_happy:
  dfi['item_orcado'] = dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'coquetel_e_happy')
for i in mixologia:
  dfi['item_orcado'] = dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'mixologia')
for i in outras_bebidas:
  dfi['item_orcado'] = dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'outras_bebidas')
for i in almoco_ou_janta:
  dfi['item_orcado'] = dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'almoco_ou_janta')
for i in cafes:
  dfi['item_orcado'] = dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'cafes')
for i in buffet:
  dfi['item_orcado'] = dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'buffet')
for i in outros_alimentos:
  dfi['item_orcado'] = dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'outros_alimentos')
for i in bolos:
  dfi['item_orcado'] = dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'bolos')
for i in pascoa:
  dfi['item_orcado'] = dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'pascoa')
for i in doces:
  dfi['item_orcado'] = dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'doces')
for i in outros_confeitaria:
  dfi['item_orcado'] = dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'outros_confeitaria')
for i in pessoas_aeb:
  dfi['item_orcado'] = dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'pessoas_aeb')
for i in outros_aeb:
  dfi['item_orcado'] = dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'outros_aeb')
for i in camisetas:
  dfi['item_orcado'] = dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'camisetas')
for i in blusas:
  dfi['item_orcado']= dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'blusas')
for i in calcados:
  dfi['item_orcado']= dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'calcados')
for i in acessorios:
  dfi['item_orcado']= dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'acessorios')
for i in bolsas:
  dfi['item_orcado']= dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'bolsas')
for i in fone:
  dfi['item_orcado']= dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'fone')
for i in item_informatica:
  dfi['item_orcado']= dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'item_informatica')
for i in cadernos_e_moleskine:
  dfi['item_orcado']= dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'cadernos_e_moleskine')
for i in item_escritorio:
  dfi['item_orcado']= dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'item_escritorio')
for i in outros_itens:
  dfi['item_orcado']= dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'outros_itens')
for i in plantas:
  dfi['item_orcado']= dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'plantas')
for i in brindes:
  dfi['item_orcado']= dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'brindes')
for i in livros:
  dfi['item_orcado']= dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'livros')
for i in transporte_de_pessoas:
  dfi['item_orcado']= dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'transporte_de_pessoas')
for i in transporte_de_itens:
  dfi['item_orcado']= dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'transporte_de_itens')
for i in local_e_espaço:
  dfi['item_orcado']= dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'local_e_espaço')
for i in mesas_e_balcoes:
  dfi['item_orcado']= dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'mesas_e_balcoes')
for i in cadeiras_e_assentos:
  dfi['item_orcado']= dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'cadeiras_e_assentos')
for i in outros_moveis:
  dfi['item_orcado']= dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'outros_moveis')
for i in equipamento_de_som:
  dfi['item_orcado']= dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'equipamento_de_som')
for i in equipamento_de_iluminação:
  dfi['item_orcado']= dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'equipamento_de_iluminação')
for i in equipamento_de_video:
  dfi['item_orcado']= dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'equipamento_de_video')
for i in outros_equipamentos:
  dfi['item_orcado']= dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'outros_equipamentos')
for i in limpeza:
  dfi['item_orcado']= dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'limpeza')
for i in seguranca:
  dfi['item_orcado']= dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'seguranca')
for i in testes_e_exames:
  dfi['item_orcado']= dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'testes_e_exames')
for i in ambulancia_e_afins:
  dfi['item_orcado']= dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'ambulancia_e_afins')
for i in bombeiro:
  dfi['item_orcado']= dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'bombeiro')
for i in ap_musical:
  dfi['item_orcado']= dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'ap_musical')
for i in ap_outros:
  dfi['item_orcado']= dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'ap_outros')
for i in jogos_e_dinamicas:
  dfi['item_orcado']= dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'jogos_e_dinamicas')
for i in staff_e_organ:
  dfi['item_orcado']= dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'staff_e_organ')
for i in recepcao_hostness:
  dfi['item_orcado'] = dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'recepcao_hostness')
for i in planejamento_e_producao:
  dfi['item_orcado'] = dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'planejamento_e_producao')
for i in traducao_e_linguas:
  dfi['item_orcado'] = dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'traducao_e_linguas')
for i in locucoes_e_narracoes:
  dfi['item_orcado'] = dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'locucoes_e_narracoes')
for i in tecnico_som:
  dfi['item_orcado'] = dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'tecnico_som')
for i in tecnico_filmagem:
  dfi['item_orcado'] = dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'tecnico_filmagem')
for i in tecnico_streaming:
  dfi['item_orcado'] = dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'tecnico_streaming')
for i in outros_tecnicos:
  dfi['item_orcado'] = dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'outros_tecnicos')
for i in banners_e_faixas:
  dfi['item_orcado'] = dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'banners_e_faixas')
for i in outros_divulgacao:
  dfi['item_orcado'] = dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'outros_divulgacao')
for i in outros_impressos:
  dfi['item_orcado'] = dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'outros_impressos')
for i in flyer_folder:
  dfi['item_orcado'] = dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'flyer_folder')
for i in cartao_e_ingresso:
  dfi['item_orcado'] = dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'cartao_e_ingresso')
for i in caixa_e_pacote:
  dfi['item_orcado'] = dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'caixa_e_pacote')
for i in kits_e_cestas:
  dfi['item_orcado'] = dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'kits_e_cestas')
for i in calcas:
  dfi['item_orcado'] = dfi['item_orcado'].where((~dfi['name'].str.contains(i)),'calcas')

bebidas= ['cerveja_e_chope','vinhos','coquetel_e_happy','mixologia','outras_bebidas']
alimentacao= ['almoco_ou_janta','cafes','buffet','outros_alimentos']
confeitaria= ['bolos','pascoa','doces','outros_confeitaria']
servicoAeB= ['pessoas_a&b','outros_a&b']
roupas_e_vestuario = ['camisetas','blusas','calcados','acessorios','bolsas','calcas']
personalizados_e_brindes = ['fone','item_informatica','cadernos_e_moleskine','item_escritorio','outros_itens','plantas','brindes','livros']
logistica = ['transporte_de_pessoas','transporte_de_itens','transporte_de_pessoas','transporte_de_itens',]
infraestrutura = ['local_e_espaço']
mobiliario =['mesas_e_balcoes','cadeiras_e_assentos','outros_moveis']
equipamentos = ['equipamento_de_som','equipamento_de_iluminacao','equipamento_de_video','outros_equipamentos']
saude = ['testes_e_exames','ambulancia_e_afins']
limpeza_e_seguranca = ['limpeza','seguranca','bombeiro']
entreterimento = ['ap_musical','ap_outros','jogos_e_dinamicas']
organizacao = ['staff_e_organ','recepcao_hostness','planejamento_e_producao']
tecnicos = ['traducao_e_linguas','locucoes_e_narracoes','tecnico_som','tecnico_filmagem','tecnico_streaming','outros_tecnicos']
graficos = ['banners_e_faixas','outros_divulgacao']
impressos =['flyer_folder','outros_impressos','cartao_e_ingresso']
kits_e_afins = ['caixa_e_pacote','kits_e_cestas']
caixa_e_pacote = ['caixa','caixinha','caixote','embalagem','pacote','box']

dfi['categoria'] = 'Outros'
for i in alimentacao:
  dfi['categoria'] = dfi['categoria'].where((~dfi['item_orcado'].str.contains(i)),'alimentacao')
for i in bebidas:
  dfi['categoria'] = dfi['categoria'].where((~dfi['item_orcado'].str.contains(i)),'bebidas')
for i in confeitaria:
  dfi['categoria'] = dfi['categoria'].where((~dfi['item_orcado'].str.contains(i)),'confeitaria')
for i in servicoAeB:
  dfi['categoria'] = dfi['categoria'].where((~dfi['item_orcado'].str.contains(i)),'servicoAeB')
for i in roupas_e_vestuario:
  dfi['categoria'] = dfi['categoria'].where((~dfi['item_orcado'].str.contains(i)),'roupas_e_vestuario')
for i in personalizados_e_brindes:
  dfi['categoria'] = dfi['categoria'].where((~dfi['item_orcado'].str.contains(i)),'personalizados_e_brindes')
for i in logistica:
  dfi['categoria'] = dfi['categoria'].where((~dfi['item_orcado'].str.contains(i)),'logistica')
for i in infraestrutura:
  dfi['categoria'] = dfi['categoria'].where((~dfi['item_orcado'].str.contains(i)),'infraestrutura')
for i in mobiliario:
  dfi['categoria'] = dfi['categoria'].where((~dfi['item_orcado'].str.contains(i)),'mobiliario')
for i in equipamentos:
  dfi['categoria'] = dfi['categoria'].where((~dfi['item_orcado'].str.contains(i)),'equipamentos')
for i in saude:
  dfi['categoria'] = dfi['categoria'].where((~dfi['item_orcado'].str.contains(i)),'saude')
for i in limpeza_e_seguranca:
  dfi['categoria'] = dfi['categoria'].where((~dfi['item_orcado'].str.contains(i)),'limpeza_e_seguranca')
for i in kits_e_afins:
  dfi['categoria'] = dfi['categoria'].where((~dfi['item_orcado'].str.contains(i)),'kits_e_afins')
for i in entreterimento:
  dfi['categoria'] = dfi['categoria'].where((~dfi['item_orcado'].str.contains(i)),'entreterimento')
for i in organizacao:
  dfi['categoria'] = dfi['categoria'].where((~dfi['item_orcado'].str.contains(i)),'organizacao')
for i in tecnicos:
  dfi['categoria'] = dfi['categoria'].where((~dfi['item_orcado'].str.contains(i)),'tecnicos')
for i in graficos:
  dfi['categoria'] = dfi['categoria'].where((~dfi['item_orcado'].str.contains(i)),'graficos')
for i in impressos:
  dfi['categoria'] = dfi['categoria'].where((~dfi['item_orcado'].str.contains(i)),'impressos')
for i in caixa_e_pacote:
  dfi['categoria'] = dfi['categoria'].where((~dfi['item_orcado'].str.contains(i)),'caixa_e_pacote')

dfi['área'] = dfi['categoria']

AeB = ['alimentacao','bebidas','confeitarua','servicoAeB']
Produtos = ['roupas_e_vestuario','personalizados_e_brindes','kits_e_afins','caixa_e_pacote']
logistica1 = ['logistica']
estrutura = ['infraestrutura','mobiliario','equipamentos']
servicos = ['saude','limpeza_e_seguranca','entreterimento','organizacao','tecnicos']
comunicacao = ['graficos','impressos']

for i in AeB:
  dfi['área'] = dfi['área'].where((~dfi['área'].str.contains(i)),'A&B')
for i in Produtos:
  dfi['área'] = dfi['área'].where((~dfi['área'].str.contains(i)),'Produtos Personalizados')
for i in logistica:
  dfi['área'] = dfi['área'].where((~dfi['área'].str.contains(i)),'Logística e transporte')
for i in estrutura:
  dfi['área'] = dfi['área'].where((~dfi['área'].str.contains(i)),'Estrutura e local')
for i in servicos:
  dfi['área'] = dfi['área'].where((~dfi['área'].str.contains(i)),'Serviços gerais e técnicos')
for i in comunicacao:
  dfi['área'] = dfi['área'].where((~dfi['área'].str.contains(i)),'Comunicações')


dfi.to_csv('df_tratado/dfi.csv')

##4.2) Criando KPI do fechamento das vendas

# Criando a coluna do que virou pedido "is_order"
df_propostas['is_order'] = df_propostas['id'].apply(lambda x: int(1) if x in df_orders['client_proposal_id'].unique() else int(0))

# Filtrando o dataset para pedidos a partir de novembro, tendo em vista que a vinculação com o número do pedido só foi implantada em novembro
df_propostas_filtrada = df_propostas[ df_propostas.updated_at >= '2021-11-01']

##4.3) Preparando DF para o modelo

dfi.drop(columns=['id','original_item_id','supplier_sale_price','duration','quantity','price_method','editable_quantity','min_quantity','max_quantity',\
                  'additional_fee_name','additional_fee','supplier_name','obs','section_id','section_name','section_position'], inplace=True)
dfi['margem'] = (dfi['buyer_price'] - dfi['sale_price'])*100/dfi['sale_price']

df_propostas_filtrada = df_propostas_filtrada.drop(columns=['admin','pax','start_date','address','expires_at','created_at'])

X = pd.merge(df_propostas_filtrada, dfi, how = 'left', left_on='id', right_on='client_proposal_id')

#Excluir as linhas onde há id dos itens nulos
X = X[ ~X['client_proposal_id'].isnull() ]
X[X['final_buyer_price'].isnull()]
media = X[X['categoria']=='kits_e_afins']['final_buyer_price'].mean()
X = X.fillna(media)

X =X.drop(columns=['id','name_x','updated_at','client_proposal_id','name_y','sale_price','buyer_price','área','item_orcado'])
X = pd.get_dummies(X, prefix=['complexity','categoria'])

y=X['is_order']
X=X.drop(columns=['is_order'])

# Salvar X e y para evitar reiniciar todo o código.

X.to_csv("df_tratado\df_X.csv", index = False)
y.to_csv("df_tratado\df_y.csv", index = False)

#5) Separando os DF's em treino e teste

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7, random_state=5)

#6) Gradiente Boosting

clf = GradientBoostingClassifier(random_state=5, criterion='squared_error', learning_rate=1, max_depth=5, max_features='auto', min_samples_split=2, n_estimators=150)
clf.fit(X_train, y_train)

#7) Salvando modelo Treinado

joblib.dump(clf, 'Modelo\modelo.pkl')