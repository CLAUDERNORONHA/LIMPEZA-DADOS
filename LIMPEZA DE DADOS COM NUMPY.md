## <font color='blue'>Estudo de Caso 1 - Limpeza e Pré-Processamento de Dados com NumPy</font>

![title](imagens/EstudoCaso1.png)


```python
# Versão da Linguagem Python
from platform import python_version
print('Versão da Linguagem Python Usada Neste Jupyter Notebook:', python_version())
```

    Versão da Linguagem Python Usada Neste Jupyter Notebook: 3.9.7
    


```python
# Para atualizar um pacote, execute o comando abaixo no terminal ou prompt de comando:
# pip install -U nome_pacote

# Para instalar a versão exata de um pacote, execute o comando abaixo no terminal ou prompt de comando:
# !pip install nome_pacote==versão_desejada

# Depois de instalar ou atualizar o pacote, reinicie o jupyter notebook.

# Instala o pacote watermark. 
# Esse pacote é usado para gravar as versões de outros pacotes usados neste jupyter notebook.
# !pip install -q -U watermark
```


```python
# Import
import numpy as np
```


```python
import warnings
warnings.filterwarnings('ignore')
```


```python
# Versões dos pacotes usados neste jupyter notebook
%reload_ext watermark
%watermark -a "Data Science Academy" --iversions
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    C:\Users\CLAUDE~1.NOR\AppData\Local\Temp/ipykernel_6740/559600613.py in <module>
          1 # Versões dos pacotes usados neste jupyter notebook
    ----> 2 get_ipython().run_line_magic('reload_ext', 'watermark')
          3 get_ipython().run_line_magic('watermark', '-a "Data Science Academy" --iversions')
    

    ~\Anaconda3\lib\site-packages\IPython\core\interactiveshell.py in run_line_magic(self, magic_name, line, _stack_depth)
       2349                 kwargs['local_ns'] = self.get_local_scope(stack_depth)
       2350             with self.builtin_trap:
    -> 2351                 result = fn(*args, **kwargs)
       2352             return result
       2353 
    

    ~\Anaconda3\lib\site-packages\decorator.py in fun(*args, **kw)
        230             if not kwsyntax:
        231                 args, kw = fix(args, kw, sig)
    --> 232             return caller(func, *(extras + args), **kw)
        233     fun.__name__ = func.__name__
        234     fun.__doc__ = func.__doc__
    

    ~\Anaconda3\lib\site-packages\IPython\core\magic.py in <lambda>(f, *a, **k)
        185     # but it's overkill for just that one bit of state.
        186     def magic_deco(arg):
    --> 187         call = lambda f, *a, **k: f(*a, **k)
        188 
        189         if callable(arg):
    

    ~\Anaconda3\lib\site-packages\IPython\core\magics\extension.py in reload_ext(self, module_str)
         61         if not module_str:
         62             raise UsageError('Missing module name.')
    ---> 63         self.shell.extension_manager.reload_extension(module_str)
    

    ~\Anaconda3\lib\site-packages\IPython\core\extensions.py in reload_extension(self, module_str)
        128                 self.loaded.add(module_str)
        129         else:
    --> 130             self.load_extension(module_str)
        131 
        132     def _call_load_ipython_extension(self, mod):
    

    ~\Anaconda3\lib\site-packages\IPython\core\extensions.py in load_extension(self, module_str)
         78             if module_str not in sys.modules:
         79                 with prepended_to_syspath(self.ipython_extension_dir):
    ---> 80                     mod = import_module(module_str)
         81                     if mod.__file__.startswith(self.ipython_extension_dir):
         82                         print(("Loading extensions from {dir} is deprecated. "
    

    ~\Anaconda3\lib\importlib\__init__.py in import_module(name, package)
        125                 break
        126             level += 1
    --> 127     return _bootstrap._gcd_import(name[level:], package, level)
        128 
        129 
    

    ~\Anaconda3\lib\importlib\_bootstrap.py in _gcd_import(name, package, level)
    

    ~\Anaconda3\lib\importlib\_bootstrap.py in _find_and_load(name, import_)
    

    ~\Anaconda3\lib\importlib\_bootstrap.py in _find_and_load_unlocked(name, import_)
    

    ModuleNotFoundError: No module named 'watermark'


https://numpy.org/doc/stable/reference/generated/numpy.set_printoptions.html


```python
# Configuração de impressão do NumPy
np.set_printoptions(suppress = True, linewidth = 200, precision = 2)
```

## Carregando o Dataset

https://numpy.org/doc/stable/reference/generated/numpy.genfromtxt.html


```python
dados = np.genfromtxt("dados/dataset1.csv", 
                      delimiter = ';', 
                      skip_header = 1, 
                      autostrip = True, 
                      encoding = 'cp1252')
```


```python
type(dados)
```




    numpy.ndarray




```python
dados.shape
```




    (10000, 14)




```python
dados.view()
```




    array([[48010226.  ,         nan,    35000.  , ...,         nan,         nan,     9452.96],
           [57693261.  ,         nan,    30000.  , ...,         nan,         nan,     4679.7 ],
           [59432726.  ,         nan,    15000.  , ...,         nan,         nan,     1969.83],
           ...,
           [50415990.  ,         nan,    10000.  , ...,         nan,         nan,     2185.64],
           [46154151.  ,         nan,         nan, ...,         nan,         nan,     3199.4 ],
           [66055249.  ,         nan,    10000.  , ...,         nan,         nan,      301.9 ]])



Observe como várias colunas estão com o tipo nan. Isso se deve a caracteres especiais no conjunto de dados e a forma como o NumPy carrega dados numéricos e do tipo string. Vamos resolver isso.

## Verificando Valores Ausentes


```python
np.isnan(dados).sum()
```




    88005



https://numpy.org/doc/stable/reference/generated/numpy.nanmax.html


```python
# Vamos retornar o maior valor + 1 ignorando valores nan
# Usaremos esse valor arbitrário para preencher os valores ausentes no momento da carga de dados de variáveis
# numéricas e depois tratamos esse valor como valor ausente
valor_coringa = np.nanmax(dados) + 1
print(valor_coringa)
```

    68616520.0
    

https://numpy.org/doc/stable/reference/generated/numpy.nanmean.html


```python
# Calculamos a média (variáveis numéricas) ignorando valores nan por coluna
# Usaremos isso para separar variáveis numéricas de variáveis do tipo string
media_ignorando_nan = np.nanmean(dados, axis = 0)
print(media_ignorando_nan)
```

    [54015809.19         nan    15273.46         nan    15311.04         nan       16.62      440.92         nan         nan         nan         nan         nan     3143.85]
    

https://numpy.org/doc/stable/reference/generated/numpy.argwhere.html

https://numpy.org/doc/stable/reference/generated/numpy.squeeze.html


```python
# Colunas do tipo string com valores ausentes
colunas_strings = np.argwhere(np.isnan(media_ignorando_nan)).squeeze()
colunas_strings
```




    array([ 1,  3,  5,  8,  9, 10, 11, 12], dtype=int64)




```python
# Colunas numéricas 
colunas_numericas = np.argwhere(np.isnan(media_ignorando_nan) == False).squeeze()
colunas_numericas
```




    array([ 0,  2,  4,  6,  7, 13], dtype=int64)



> Importamos novamente o dataset, separando colunas do tipo string de colunas numéricas.


```python
# Carrega as colunas do tipo string
arr_strings = np.genfromtxt("dados/dataset1.csv",
                            delimiter = ';',
                            skip_header = 1,
                            autostrip = True, 
                            usecols = colunas_strings,
                            dtype = str, 
                            encoding = 'cp1252')
```


```python
arr_strings
```




    array([['May-15', 'Current', '36 months', ..., 'Verified', 'https://www.lendingclub.com/browse/loanDetail.action?loan_id=48010226', 'CA'],
           ['', 'Current', '36 months', ..., 'Source Verified', 'https://www.lendingclub.com/browse/loanDetail.action?loan_id=57693261', 'NY'],
           ['Sep-15', 'Current', '36 months', ..., 'Verified', 'https://www.lendingclub.com/browse/loanDetail.action?loan_id=59432726', 'PA'],
           ...,
           ['Jun-15', 'Current', '36 months', ..., 'Source Verified', 'https://www.lendingclub.com/browse/loanDetail.action?loan_id=50415990', 'CA'],
           ['Apr-15', 'Current', '36 months', ..., 'Source Verified', 'https://www.lendingclub.com/browse/loanDetail.action?loan_id=46154151', 'OH'],
           ['Dec-15', 'Current', '36 months', ..., '', 'https://www.lendingclub.com/browse/loanDetail.action?loan_id=66055249', 'IL']], dtype='<U69')




```python
# Carrega as colunas do tipo numérico preenchendo os valores ausentes
arr_numeric = np.genfromtxt("dados/dataset1.csv",
                            delimiter = ';',
                            autostrip = True,
                            skip_header = 1,
                            usecols = colunas_numericas,
                            filling_values = valor_coringa, 
                            encoding = 'cp1252')
```


```python
arr_numeric
```




    array([[48010226.  ,    35000.  ,    35000.  ,       13.33,     1184.86,     9452.96],
           [57693261.  ,    30000.  ,    30000.  , 68616520.  ,      938.57,     4679.7 ],
           [59432726.  ,    15000.  ,    15000.  , 68616520.  ,      494.86,     1969.83],
           ...,
           [50415990.  ,    10000.  ,    10000.  , 68616520.  , 68616520.  ,     2185.64],
           [46154151.  , 68616520.  ,    10000.  ,       16.55,      354.3 ,     3199.4 ],
           [66055249.  ,    10000.  ,    10000.  , 68616520.  ,      309.97,      301.9 ]])



> Agora extraímos os nomes das colunas.


```python
# Carrega os nomes das colunas
arr_nomes_colunas = np.genfromtxt("dados/dataset1.csv",
                                  delimiter = ';',
                                  autostrip = True,
                                  skip_footer = dados.shape[0],
                                  dtype = str, 
                                  encoding = 'cp1252')
```


```python
arr_nomes_colunas
```




    array(['id', 'issue_d', 'loan_amnt', 'loan_status', 'funded_amnt', 'term', 'int_rate', 'installment', 'grade', 'sub_grade', 'verification_status', 'url', 'addr_state', 'total_pymnt'], dtype='<U19')




```python
# Separa cabeçalho de colunas numéricas e string
header_strings, header_numeric = arr_nomes_colunas[colunas_strings], arr_nomes_colunas[colunas_numericas]
```


```python
header_strings
```




    array(['issue_d', 'loan_status', 'term', 'grade', 'sub_grade', 'verification_status', 'url', 'addr_state'], dtype='<U19')




```python
header_numeric
```




    array(['id', 'loan_amnt', 'funded_amnt', 'int_rate', 'installment', 'total_pymnt'], dtype='<U19')



## Função de Checkpoint

**Checkpoint 1**
Serve par anão começar tudo do zero.
Vamos criar uma função de checkpoint para salvar os resultados intermédiários.


```python
# Função
def checkpoint(file_name, checkpoint_header, checkpoint_data):
    np.savez(file_name, header = checkpoint_header, data = checkpoint_data)
    checkpoint_variable = np.load(file_name + ".npz")
    return(checkpoint_variable)
```


```python
checkpoint_inicial = checkpoint("dados/Checkpoint-Inicial", header_strings, arr_strings)
```


```python
checkpoint_inicial['data']

#Os dados ainda não foram limpos.
```




    array([['May-15', 'Current', '36 months', ..., 'Verified', 'https://www.lendingclub.com/browse/loanDetail.action?loan_id=48010226', 'CA'],
           ['', 'Current', '36 months', ..., 'Source Verified', 'https://www.lendingclub.com/browse/loanDetail.action?loan_id=57693261', 'NY'],
           ['Sep-15', 'Current', '36 months', ..., 'Verified', 'https://www.lendingclub.com/browse/loanDetail.action?loan_id=59432726', 'PA'],
           ...,
           ['Jun-15', 'Current', '36 months', ..., 'Source Verified', 'https://www.lendingclub.com/browse/loanDetail.action?loan_id=50415990', 'CA'],
           ['Apr-15', 'Current', '36 months', ..., 'Source Verified', 'https://www.lendingclub.com/browse/loanDetail.action?loan_id=46154151', 'OH'],
           ['Dec-15', 'Current', '36 months', ..., '', 'https://www.lendingclub.com/browse/loanDetail.action?loan_id=66055249', 'IL']], dtype='<U69')




```python
np.array_equal(checkpoint_inicial['data'], arr_strings)
```




    True



## Manipulando as Colunas do Tipo String


```python
header_strings
```




    array(['issue_d', 'loan_status', 'term', 'grade', 'sub_grade', 'verification_status', 'url', 'addr_state'], dtype='<U19')




```python
# Vamos ajustar o nome da coluna issue_d para facilitar a identificação da coluna
header_strings[0] = "issue_date"

#Modificar os nomes das colunas 
```


```python
header_strings
```




    array(['issue_date', 'loan_status', 'term', 'grade', 'sub_grade', 'verification_status', 'url', 'addr_state'], dtype='<U19')




```python
arr_strings
```




    array([['May-15', 'Current', '36 months', ..., 'Verified', 'https://www.lendingclub.com/browse/loanDetail.action?loan_id=48010226', 'CA'],
           ['', 'Current', '36 months', ..., 'Source Verified', 'https://www.lendingclub.com/browse/loanDetail.action?loan_id=57693261', 'NY'],
           ['Sep-15', 'Current', '36 months', ..., 'Verified', 'https://www.lendingclub.com/browse/loanDetail.action?loan_id=59432726', 'PA'],
           ...,
           ['Jun-15', 'Current', '36 months', ..., 'Source Verified', 'https://www.lendingclub.com/browse/loanDetail.action?loan_id=50415990', 'CA'],
           ['Apr-15', 'Current', '36 months', ..., 'Source Verified', 'https://www.lendingclub.com/browse/loanDetail.action?loan_id=46154151', 'OH'],
           ['Dec-15', 'Current', '36 months', ..., '', 'https://www.lendingclub.com/browse/loanDetail.action?loan_id=66055249', 'IL']], dtype='<U69')



### Pré-Processamento da Variável issue_date com Label Encoding


```python
# Extrai os valores únicos da variável
np.unique(arr_strings[:,0])#variavel 0

#Nos meses eles colocaram o 15, talvez uma forma de organização. Mas vamos tirar 
#esse número 15


#Apareceu um valor vazio no canto esquerdo, esse valor vazio são dados ausentes.
```




    array(['', 'Apr-15', 'Aug-15', 'Dec-15', 'Feb-15', 'Jan-15', 'Jul-15', 'Jun-15', 'Mar-15', 'May-15', 'Nov-15', 'Oct-15', 'Sep-15'], dtype='<U69')




```python
# Vamos remover o sufixo -15 e converter em um array de strings
arr_strings[:,0] = np.chararray.strip(arr_strings[:,0], "-15")
```


```python
# Extrai os valores únicos da variável
np.unique(arr_strings[:,0])
```




    array(['', 'Apr', 'Aug', 'Dec', 'Feb', 'Jan', 'Jul', 'Jun', 'Mar', 'May', 'Nov', 'Oct', 'Sep'], dtype='<U69')




```python
# Criamos um array com os meses (incluindo um elemento como vazio para o que estiver em branco)
meses = np.array(['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
```


```python
# Loop para converter os nomes dos meses em valores numéricos
# Chamamos isso de label encoding
for i in range(13):
        arr_strings[:,0] = np.where(arr_strings[:,0] == meses[i], i, arr_strings[:,0])
```


```python
np.unique(arr_strings[:,0])
```




    array(['0', '1', '10', '11', '12', '2', '3', '4', '5', '6', '7', '8', '9'], dtype='<U69')



### Pré-Processamento da Variável loan_status com Binarização


```python
header_strings
```




    array(['issue_date', 'loan_status', 'term', 'grade', 'sub_grade', 'verification_status', 'url', 'addr_state'], dtype='<U19')




```python
# Extrai os valores únicos da variável
np.unique(arr_strings[:,1])#o 1 significa a segunda variavel da nossa array
```




    array(['', 'Charged Off', 'Current', 'Default', 'Fully Paid', 'In Grace Period', 'Issued', 'Late (16-30 days)', 'Late (31-120 days)'], dtype='<U69')




```python
# Número de elementos
np.unique(arr_strings[:,1]).size
```




    9




```python

# Criamos um array com apenas 3 status
status_bad = np.array(['', 'Charged Off', 'Default', 'Late (31-120 days)'])
```


```python
# Checamos agora os valores da variável e comparamos com o array anterior convertendo a variável para valores binários
# Chamamos isso de binarização
arr_strings[:,1] = np.where(np.isin(arr_strings[:,1], status_bad),0,1)
```


```python
# Extrai os valores únicos da variável
np.unique(arr_strings[:,1])
```




    array(['0', '1'], dtype='<U69')



### Pré-Processamento da Variável term com Limpeza de String


```python
header_strings
```




    array(['issue_date', 'loan_status', 'term', 'grade', 'sub_grade', 'verification_status', 'url', 'addr_state'], dtype='<U19')




```python
# Extrai os valores únicos da variável
np.unique(arr_strings[:,2])
```




    array(['', '36 months', '60 months'], dtype='<U69')




```python
# Removemos a palavra months (observe o espaço antes da palavra)
arr_strings[:,2] = np.chararray.strip(arr_strings[:,2], " months")
arr_strings[:,2]
```




    array(['36', '36', '36', ..., '36', '36', '36'], dtype='<U69')




```python
# Mudamos o título da variável
header_strings[2] = "term_months"
```


```python
# Substituímos os valores ausentes pelo maior valor, em nosso caso 60
arr_strings[:,2] = np.where(arr_strings[:,2] == '', '60', arr_strings[:,2])
```


```python
arr_strings[:,2]
```




    array(['36', '36', '36', ..., '36', '36', '36'], dtype='<U69')




```python
# Extrai os valores únicos da variável
np.unique(arr_strings[:,2])
```




    array(['36', '60'], dtype='<U69')



### Pré-Processamento das  Variáveis grade e sub_grade com Dicionário (Tipo de Label Encoding)


```python
header_strings
```




    array(['issue_date', 'loan_status', 'term_months', 'grade', 'sub_grade', 'verification_status', 'url', 'addr_state'], dtype='<U19')




```python
# Extrai os valores únicos da variável
np.unique(arr_strings[:,3])


#Notas dadas pelos clientes
```




    array(['', 'A', 'B', 'C', 'D', 'E', 'F', 'G'], dtype='<U69')




```python
# Extrai os valores únicos da variável

np.unique(arr_strings[:,4])


#Sub-categoria das notas.
#Temosos mesmos niveis de informação, será que é uma boa manter as duas?

#Temos que escolher a mlhor informação para deixar dentro do dataset
```




    array(['', 'A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5', 'C1', 'C2', 'C3', 'C4', 'C5', 'D1', 'D2', 'D3', 'D4', 'D5', 'E1', 'E2', 'E3', 'E4', 'E5', 'F1', 'F2', 'F3', 'F4', 'F5', 'G1',
           'G2', 'G3', 'G4', 'G5'], dtype='<U69')



Vamos ajustar a variável sub_grade.


```python
np.unique(arr_strings[:,3])

#tem valores ausentes, devemos tirar esse valor ausente
```




    array(['', 'A', 'B', 'C', 'D', 'E', 'F', 'G'], dtype='<U69')




```python
np.unique(arr_strings[:,3])[1:]

#Tiramos os valores ausentes

```




    array(['A', 'B', 'C', 'D', 'E', 'F', 'G'], dtype='<U69')




```python
# Loop para ajuste da variável sub_grade
for i in np.unique(arr_strings[:,3])[1:]:
    arr_strings[:,4] = np.where((arr_strings[:,4] == '') & (arr_strings[:,3] == i), i + '5', arr_strings[:,4])
    
    
    
    
    
    
```


```python
# Retorna categorias e suas respectivas contagens
np.unique(arr_strings[:,4], return_counts = True)
```




    (array(['', 'A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5', 'C1', 'C2', 'C3', 'C4', 'C5', 'D1', 'D2', 'D3', 'D4', 'D5', 'E1', 'E2', 'E3', 'E4', 'E5', 'F1', 'F2', 'F3', 'F4', 'F5', 'G1',
            'G2', 'G3', 'G4', 'G5'], dtype='<U69'),
     array([  9, 285, 278, 239, 323, 592, 509, 517, 530, 553, 633, 629, 567, 586, 564, 577, 391, 267, 250, 255, 288, 235, 162, 171, 139, 160,  94,  52,  34,  43,  24,  19,  10,   3,   7,   5], dtype=int64))




```python
# Substituímos valores ausentes por uma H1nova categoria
arr_strings[:,4] = np.where(arr_strings[:,4] == '', 'H1', arr_strings[:,4])
```


```python
# Extrai os valores únicos da variável
np.unique(arr_strings[:,4])
```




    array(['A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5', 'C1', 'C2', 'C3', 'C4', 'C5', 'D1', 'D2', 'D3', 'D4', 'D5', 'E1', 'E2', 'E3', 'E4', 'E5', 'F1', 'F2', 'F3', 'F4', 'F5', 'G1', 'G2',
           'G3', 'G4', 'G5', 'H1'], dtype='<U69')



Vamos remover a variável grade.


```python
# Não precisamos mais da variável grade. Podemos removê-la.
arr_strings = np.delete(arr_strings, 3, axis = 1)
```


```python
# Nova variável na coluna de índice 3
arr_strings[:,3]
```




    array(['C3', 'A5', 'B5', ..., 'A5', 'D2', 'A4'], dtype='<U69')




```python
# Não podemos esquecer de remover a coluna do array de nomes de colunas
header_strings = np.delete(header_strings, 3)
```


```python
# Nova variável na coluna de índice 3
header_strings[3]
```




    'sub_grade'



Por fim, convertemos a variável sub_grade para sua representação numérica.


```python
# Extrai os valores únicos da variável
np.unique(arr_strings[:,3])
```




    array(['A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5', 'C1', 'C2', 'C3', 'C4', 'C5', 'D1', 'D2', 'D3', 'D4', 'D5', 'E1', 'E2', 'E3', 'E4', 'E5', 'F1', 'F2', 'F3', 'F4', 'F5', 'G1', 'G2',
           'G3', 'G4', 'G5', 'H1'], dtype='<U69')




```python
# Cria uma lista de chaves, criando um dicionario
keys = list(np.unique(arr_strings[:,3]))     
keys[0]
```




    'A1'




```python
# Cria uma lista de valores
values = list(range(1, np.unique(arr_strings[:,3]).shape[0] + 1)) 
values[0]

#criando categorias 
```




    1




```python
# Criamos então o dicionário
dict_sub_grade = dict(zip(keys, values))
```


```python
dict_sub_grade
```




    {'A1': 1,
     'A2': 2,
     'A3': 3,
     'A4': 4,
     'A5': 5,
     'B1': 6,
     'B2': 7,
     'B3': 8,
     'B4': 9,
     'B5': 10,
     'C1': 11,
     'C2': 12,
     'C3': 13,
     'C4': 14,
     'C5': 15,
     'D1': 16,
     'D2': 17,
     'D3': 18,
     'D4': 19,
     'D5': 20,
     'E1': 21,
     'E2': 22,
     'E3': 23,
     'E4': 24,
     'E5': 25,
     'F1': 26,
     'F2': 27,
     'F3': 28,
     'F4': 29,
     'F5': 30,
     'G1': 31,
     'G2': 32,
     'G3': 33,
     'G4': 34,
     'G5': 35,
     'H1': 36}




```python
# Loop para substituir a string com as categorias pela representação numérica (frequência)
for i in np.unique(arr_strings[:,3]):
        arr_strings[:,3] = np.where(arr_strings[:,3] == i, dict_sub_grade[i], arr_strings[:,3])
```


```python
# Extrai os valores únicos da variável
np.unique(arr_strings[:,3])
```




    array(['1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36', '4', '5', '6',
           '7', '8', '9'], dtype='<U69')



### Pré-Processamento da Variável Verification status com Binarização


```python
# Lista com os nomes das variáveis
header_strings
```




    array(['issue_date', 'loan_status', 'term_months', 'sub_grade', 'verification_status', 'url', 'addr_state'], dtype='<U19')




```python
# Extrai os valores únicos da variável
np.unique(arr_strings[:,4])
```




    array(['', 'Not Verified', 'Source Verified', 'Verified'], dtype='<U69')




```python
# Usamos a binarização nesta variável
arr_strings[:,4] = np.where((arr_strings[:,4] == '') | (arr_strings[:,4] == 'Not Verified'), 0, 1)
```


```python
# Extrai os valores únicos da variável
np.unique(arr_strings[:,4])
```




    array(['1'], dtype='<U69')



### Pré-Processamento da Variável url com Extração de ID


```python
# Visualiza amostra dos dados
arr_strings[:,5]


#temos as URLS 
```




    array(['https://www.lendingclub.com/browse/loanDetail.action?loan_id=48010226', 'https://www.lendingclub.com/browse/loanDetail.action?loan_id=57693261',
           'https://www.lendingclub.com/browse/loanDetail.action?loan_id=59432726', ..., 'https://www.lendingclub.com/browse/loanDetail.action?loan_id=50415990',
           'https://www.lendingclub.com/browse/loanDetail.action?loan_id=46154151', 'https://www.lendingclub.com/browse/loanDetail.action?loan_id=66055249'], dtype='<U69')




```python
# Extraímos o id ao final de cada url
np.chararray.strip(arr_strings[:,5], "https://www.lendingclub.com/browse/loanDetail.action?loan_id=")
```




    chararray(['48010226', '57693261', '59432726', ..., '50415990', '46154151', '66055249'], dtype='<U69')




```python
# Substituímos a url pelo valor do id na url
arr_strings[:,5] = np.chararray.strip(arr_strings[:,5], 
                                      "https://www.lendingclub.com/browse/loanDetail.action?loan_id=")
```


```python
# Convertemos o tipo para int32
arr_strings[:,5].astype(dtype = np.int32)
```




    array([48010226, 57693261, 59432726, ..., 50415990, 46154151, 66055249])




```python
# Parece que esse id está presente na primeira coluna do conjunto de dados.
# Vamos converter para int 32 e comparar
arr_numeric[:,0].astype(dtype = np.int32)
```




    array([48010226, 57693261, 59432726, ..., 50415990, 46154151, 66055249])




```python
np.array_equal(arr_numeric[:,0].astype(dtype = np.int32), arr_strings[:,5].astype(dtype = np.int32))
```




    True



Sim, é a mesma informação. Vamos então remover uma das colunas.


```python
# Remove do array de dados
arr_strings = np.delete(arr_strings, 5, axis = 1)
```


```python
# Remove do array de nome de coluna
header_strings = np.delete(header_strings, 5)
```


```python
# Nova coluna no índice 5
arr_strings[:,5]
```




    array(['CA', 'NY', 'PA', ..., 'CA', 'OH', 'IL'], dtype='<U69')




```python
# Nova lista de colunas
header_strings
```




    array(['issue_date', 'loan_status', 'term_months', 'sub_grade', 'verification_status', 'addr_state'], dtype='<U19')




```python
# Coluna id
arr_numeric[:,0]
```




    array([48010226., 57693261., 59432726., ..., 50415990., 46154151., 66055249.])




```python
# Coluna id agora faz parte do array de numéricos
header_numeric
```




    array(['id', 'loan_amnt', 'funded_amnt', 'int_rate', 'installment', 'total_pymnt'], dtype='<U19')



### Pré-Processamento da Variável address com Categorização


```python
header_strings
```




    array(['issue_date', 'loan_status', 'term_months', 'sub_grade', 'verification_status', 'addr_state'], dtype='<U19')




```python
# Vamos ajustar o nome da coluna, renomera para endereço porque está em padrao EUA
header_strings[5] = "state_address"
```

https://numpy.org/doc/stable/reference/generated/numpy.argsort.html


```python
# Extrai nomes e contagens
states_names, states_count = np.unique(arr_strings[:,5], return_counts = True)
```


```python
# Ordena em ordem descrescente
states_count_sorted = np.argsort(-states_count)
```


```python
# Imprime o resultado
states_names[states_count_sorted], states_count[states_count_sorted]
```




    (array(['CA', 'NY', 'TX', 'FL', '', 'IL', 'NJ', 'GA', 'PA', 'OH', 'MI', 'NC', 'VA', 'MD', 'AZ', 'WA', 'MA', 'CO', 'MO', 'MN', 'IN', 'WI', 'CT', 'TN', 'NV', 'AL', 'LA', 'OR', 'SC', 'KY', 'KS', 'OK',
            'UT', 'AR', 'MS', 'NH', 'NM', 'WV', 'HI', 'RI', 'MT', 'DE', 'DC', 'WY', 'AK', 'NE', 'SD', 'VT', 'ND', 'ME'], dtype='<U69'),
     array([1336,  777,  758,  690,  500,  389,  341,  321,  320,  312,  267,  261,  242,  222,  220,  216,  210,  201,  160,  156,  152,  148,  143,  143,  130,  119,  116,  108,  107,   84,   84,   83,
              74,   74,   61,   58,   57,   49,   44,   40,   28,   27,   27,   27,   26,   25,   24,   17,   16,   10], dtype=int64))




```python
# Substituímos valores ausentes por zero
arr_strings[:,5] = np.where(arr_strings[:,5] == '', 0, arr_strings[:,5])
```

Vamos separar os estados por regiões. Referência:
https://www2.census.gov/geo/pdfs/maps-data/maps/reference/us_regdiv.pdf


```python
# Separamos os estados por regiões
states_west = np.array(['WA', 'OR','CA','NV','ID','MT', 'WY','UT','CO', 'AZ','NM','HI','AK'])
states_south = np.array(['TX','OK','AR','LA','MS','AL','TN','KY','FL','GA','SC','NC','VA','WV','MD','DE','DC'])
states_midwest = np.array(['ND','SD','NE','KS','MN','IA','MO','WI','IL','IN','MI','OH'])
states_east = np.array(['PA','NY','NJ','CT','MA','VT','NH','ME','RI'])

#Modificamos para as regiões dos estados dos EUA, por causa do trabalho
```


```python
# Agora substituímos cada estado pelo id da sua região
arr_strings[:,5] = np.where(np.isin(arr_strings[:,5], states_west), 1, arr_strings[:,5])
arr_strings[:,5] = np.where(np.isin(arr_strings[:,5], states_south), 2, arr_strings[:,5])
arr_strings[:,5] = np.where(np.isin(arr_strings[:,5], states_midwest), 3, arr_strings[:,5])
arr_strings[:,5] = np.where(np.isin(arr_strings[:,5], states_east), 4, arr_strings[:,5])
```


```python
# Extrai os valores úinicos
np.unique(arr_strings[:,5])

#O 0 indica uma região que existe
```




    array(['0', '1', '2', '3', '4'], dtype='<U69')



**<h1>Você pode modificar os dados, mas não pode modificar a informação!**</h1>

## Convertendo o Array

Nosso array de strings agora é um array numérico. Vamos ajustar o tipo de dado.


```python
arr_strings
```




    array([['5', '1', '36', '13', '1', '1'],
           ['0', '1', '36', '5', '1', '4'],
           ['9', '1', '36', '10', '1', '4'],
           ...,
           ['6', '1', '36', '5', '1', '1'],
           ['4', '1', '36', '17', '1', '3'],
           ['12', '1', '36', '4', '1', '3']], dtype='<U69')




```python
#Vamos converter par ao tipo Inteiro
arr_strings = arr_strings.astype(int)
```


```python
arr_strings
```




    array([[ 5,  1, 36, 13,  1,  1],
           [ 0,  1, 36,  5,  1,  4],
           [ 9,  1, 36, 10,  1,  4],
           ...,
           [ 6,  1, 36,  5,  1,  1],
           [ 4,  1, 36, 17,  1,  3],
           [12,  1, 36,  4,  1,  3]])




```python
arr_strings.dtype
```




    dtype('int32')



## Checkpoint com Variáveis do Tipo String Limpas e Pré-Processadas

**Checkpoint 2**

Concluída a primeira parte, vamos gravar o checkpooint.


```python
#Salvar o projeto 

checkpoint_strings = checkpoint("dados/Checkpoint-Strings", header_strings, arr_strings)
```


```python
checkpoint_strings["header"]
```




    array(['issue_date', 'loan_status', 'term_months', 'sub_grade', 'verification_status', 'state_address'], dtype='<U19')




```python
checkpoint_strings["data"]
```




    array([[ 5,  1, 36, 13,  1,  1],
           [ 0,  1, 36,  5,  1,  4],
           [ 9,  1, 36, 10,  1,  4],
           ...,
           [ 6,  1, 36,  5,  1,  1],
           [ 4,  1, 36, 17,  1,  3],
           [12,  1, 36,  4,  1,  3]])




```python
np.array_equal(checkpoint_strings['data'], arr_strings)
```




    True



## Manipulando Colunas Numéricas


```python
# Visualiza os dados
arr_numeric
```




    array([[48010226.  ,    35000.  ,    35000.  ,       13.33,     1184.86,     9452.96],
           [57693261.  ,    30000.  ,    30000.  , 68616520.  ,      938.57,     4679.7 ],
           [59432726.  ,    15000.  ,    15000.  , 68616520.  ,      494.86,     1969.83],
           ...,
           [50415990.  ,    10000.  ,    10000.  , 68616520.  , 68616520.  ,     2185.64],
           [46154151.  , 68616520.  ,    10000.  ,       16.55,      354.3 ,     3199.4 ],
           [66055249.  ,    10000.  ,    10000.  , 68616520.  ,      309.97,      301.9 ]])




```python
# Nomes das colunas
header_numeric
```




    array(['id', 'loan_amnt', 'funded_amnt', 'int_rate', 'installment', 'total_pymnt'], dtype='<U19')




```python
# Não temos valor ausente, pois ao carregar os dados substituímos por um valor arbitrário
np.isnan(arr_numeric).sum()
```




    0




```python
valor_coringa
```




    68616520.0




```python
# Podemos checar se uma coluna foi preenchida com o valor coringa
np.isin(arr_numeric[:,0], valor_coringa)


```




    array([False, False, False, ..., False, False, False])




```python
# Podemos checar se uma coluna foi preenchida com o valor coringa
np.isin(arr_numeric[:,0], valor_coringa).sum()
```




    0



Vamos criar um array de estatísticas, especificamente valor mínimo, máximo e média de cada variável. Usaremos isso noo tratamento de valores ausentes (preenchidos com o valor coringa).


```python
#Criando um array de estatistica

# Criamos um array com valor mínimo, média e valor máximo ignorando nan
# Usaremos isso no tratamento de valores ausentes
arr_stats = np.array([np.nanmin(dados, axis = 0), media_ignorando_nan, np.nanmax(dados, axis = 0)])
```


```python
print(arr_stats)
```

    [[  373332.           nan     1000.           nan     1000.           nan        6.         31.42         nan         nan         nan         nan         nan        0.  ]
     [54015809.19         nan    15273.46         nan    15311.04         nan       16.62      440.92         nan         nan         nan         nan         nan     3143.85]
     [68616519.           nan    35000.           nan    35000.           nan       28.99     1372.97         nan         nan         nan         nan         nan    41913.62]]
    


```python
arr_stats[:, colunas_numericas]
```




    array([[  373332.  ,     1000.  ,     1000.  ,        6.  ,       31.42,        0.  ],
           [54015809.19,    15273.46,    15311.04,       16.62,      440.92,     3143.85],
           [68616519.  ,    35000.  ,    35000.  ,       28.99,     1372.97,    41913.62]])



### Pré-Processamento da Variável funded_amnt


```python
# Visualiza os dados
arr_numeric[:,2]
```




    array([35000., 30000., 15000., ..., 10000., 10000., 10000.])




```python
arr_stats[0, colunas_numericas[2]]
```




    1000.0




```python
# Ajustamos o conteúdo da coluna
arr_numeric[:,2] = np.where(arr_numeric[:,2] == valor_coringa, arr_stats[0, colunas_numericas[2]], arr_numeric[:,2])
```


```python
arr_numeric[:,2]
```




    array([35000., 30000., 15000., ..., 10000., 10000., 10000.])



### Pré-Processamento das Variáveis loan_amnt, int_rate, installment e total_pymnt


```python
# Nomes das colunas
header_numeric
```




    array(['id', 'loan_amnt', 'funded_amnt', 'int_rate', 'installment', 'total_pymnt'], dtype='<U19')




```python
# Loop para substituir o valor ausente (valor_coringa) pelos valores do array de estatísticas
for i in [1,3,4,5]:
    arr_numeric[:,i] = np.where(arr_numeric[:,i] == valor_coringa, 
                                arr_stats[2, colunas_numericas[i]], 
                                arr_numeric[:,i])
```


```python
arr_numeric
```




    array([[48010226.  ,    35000.  ,    35000.  ,       13.33,     1184.86,     9452.96],
           [57693261.  ,    30000.  ,    30000.  ,       28.99,      938.57,     4679.7 ],
           [59432726.  ,    15000.  ,    15000.  ,       28.99,      494.86,     1969.83],
           ...,
           [50415990.  ,    10000.  ,    10000.  ,       28.99,     1372.97,     2185.64],
           [46154151.  ,    35000.  ,    10000.  ,       16.55,      354.3 ,     3199.4 ],
           [66055249.  ,    10000.  ,    10000.  ,       28.99,      309.97,      301.9 ]])



### Trabalhando com o Segundo Dataset

Vamos carregar os dados de cotação USD - EURO. Cada linha do dataset corresponde à taxa de câmbio para um mês em um ano.


```python
# Carrega o segundo dataset
dados_cot = np.genfromtxt("dados/dataset2.csv", 
                          delimiter = ',', 
                          autostrip = True, 
                          skip_header = 1, 
                          usecols = 3)
```


```python
# Visualiza
dados_cot
```




    array([1.13, 1.12, 1.08, 1.11, 1.1 , 1.12, 1.09, 1.13, 1.13, 1.1 , 1.06, 1.09])




```python
# Nomes de colunas
header_strings
```




    array(['issue_date', 'loan_status', 'term_months', 'sub_grade', 'verification_status', 'state_address'], dtype='<U19')




```python
# Dados
arr_strings
```




    array([[ 5,  1, 36, 13,  1,  1],
           [ 0,  1, 36,  5,  1,  4],
           [ 9,  1, 36, 10,  1,  4],
           ...,
           [ 6,  1, 36,  5,  1,  1],
           [ 4,  1, 36, 17,  1,  3],
           [12,  1, 36,  4,  1,  3]])




```python
# A coluna 0 do array de strings é o mês
arr_strings[:,0]
```




    array([ 5,  0,  9, ...,  6,  4, 12])




```python
# Vamos atribuir a coluna de mês à variável chamada exchange_rate
exchange_rate = arr_strings[:,0]
```


```python
exchange_rate
```




    array([ 5,  0,  9, ...,  6,  4, 12])




```python
# Loop para preencher a variável exchange_rate com a taxa correspondente ao mês
# Usamos dados_cot[i - 1] devido a forma como carregamos os meses para comportar o zero
for i in range(1,13):
    exchange_rate = np.where(exchange_rate == i, dados_cot[i - 1], exchange_rate)    
```


```python
exchange_rate
```




    array([1.1 , 0.  , 1.13, ..., 1.12, 1.11, 1.09])




```python
# Onde a taxa de câmbio estiver com zero substituímos pela média
exchange_rate = np.where(exchange_rate == 0, np.mean(dados_cot), exchange_rate)
```


```python
exchange_rate
```




    array([1.1 , 1.11, 1.13, ..., 1.12, 1.11, 1.09])




```python
exchange_rate.shape
```




    (10000,)




```python
arr_numeric.shape
```




    (10000, 6)




```python
exchange_rate = np.reshape(exchange_rate, (10000,1))
```

https://numpy.org/doc/stable/reference/generated/numpy.hstack.html


```python
# Concatenação dos arrays
arr_numeric = np.hstack((arr_numeric, exchange_rate))
```


```python
# Inclui o nome da coluna no array de nomes de colunas
header_numeric = np.concatenate((header_numeric, np.array(['exchange_rate'])))
```


```python
header_numeric
```




    array(['id', 'loan_amnt', 'funded_amnt', 'int_rate', 'installment', 'total_pymnt', 'exchange_rate', 'exchange_rate'], dtype='<U19')



Vamos criar colunas para as taxas de câmbio em USD e EURO.


```python
header_numeric
```




    array(['id', 'loan_amnt', 'funded_amnt', 'int_rate', 'installment', 'total_pymnt', 'exchange_rate', 'exchange_rate'], dtype='<U19')




```python
# Colunas em USD
columns_dollar = np.array([1,2,4,5])
```


```python
# Visualiza
arr_numeric[:,6]
```




    array([1.1 , 1.11, 1.13, ..., 1.12, 1.11, 1.09])




```python
# Shape
arr_numeric.shape
```




    (10000, 7)




```python
# Loop pelas colunas USD para aplicar a taxa de conversão para EURO
for i in columns_dollar:
    arr_numeric = np.hstack((arr_numeric, np.reshape(arr_numeric[:,i] / arr_numeric[:,6], (10000,1))))
```


```python
# Shape
arr_numeric.shape
```




    (10000, 11)




```python
# Visualiza
arr_numeric
```




    array([[48010226.  ,    35000.  ,    35000.  , ...,    31933.3 ,     1081.04,     8624.69],
           [57693261.  ,    30000.  ,    30000.  , ...,    27132.46,      848.86,     4232.39],
           [59432726.  ,    15000.  ,    15000.  , ...,    13326.3 ,      439.64,     1750.04],
           ...,
           [50415990.  ,    10000.  ,    10000.  , ...,     8910.3 ,     1223.36,     1947.47],
           [46154151.  ,    35000.  ,    10000.  , ...,     8997.4 ,      318.78,     2878.63],
           [66055249.  ,    10000.  ,    10000.  , ...,     9145.8 ,      283.49,      276.11]])



Vamos expandir o cabeçalho com as novas colunas.


```python
header_additional = np.array([column_name + '_EUR' for column_name in header_numeric[columns_dollar]])
```


```python
header_additional
```




    array(['loan_amnt_EUR', 'funded_amnt_EUR', 'installment_EUR', 'total_pymnt_EUR'], dtype='<U15')




```python
header_numeric = np.concatenate((header_numeric, header_additional))
```


```python
header_numeric
```




    array(['id', 'loan_amnt', 'funded_amnt', 'int_rate', 'installment', 'total_pymnt', 'exchange_rate', 'exchange_rate', 'loan_amnt_EUR', 'funded_amnt_EUR', 'installment_EUR', 'total_pymnt_EUR'],
          dtype='<U19')




```python
header_numeric[columns_dollar] = np.array([column_name + '_USD' for column_name in header_numeric[columns_dollar]])
```


```python
header_numeric
```




    array(['id', 'loan_amnt_USD', 'funded_amnt_USD', 'int_rate', 'installment_USD', 'total_pymnt_USD', 'exchange_rate', 'exchange_rate', 'loan_amnt_EUR', 'funded_amnt_EUR', 'installment_EUR',
           'total_pymnt_EUR'], dtype='<U19')




```python
columns_index_order = [0,1,7,2,8,3,4,9,5,10,6]
```


```python
header_numeric = header_numeric[columns_index_order]
```


```python
arr_numeric
```




    array([[48010226.  ,    35000.  ,    35000.  , ...,    31933.3 ,     1081.04,     8624.69],
           [57693261.  ,    30000.  ,    30000.  , ...,    27132.46,      848.86,     4232.39],
           [59432726.  ,    15000.  ,    15000.  , ...,    13326.3 ,      439.64,     1750.04],
           ...,
           [50415990.  ,    10000.  ,    10000.  , ...,     8910.3 ,     1223.36,     1947.47],
           [46154151.  ,    35000.  ,    10000.  , ...,     8997.4 ,      318.78,     2878.63],
           [66055249.  ,    10000.  ,    10000.  , ...,     9145.8 ,      283.49,      276.11]])




```python
arr_numeric = arr_numeric[:,columns_index_order]
```

### Pré-Processamento da Variável int_rate


```python
header_numeric
```




    array(['id', 'loan_amnt_USD', 'exchange_rate', 'funded_amnt_USD', 'loan_amnt_EUR', 'int_rate', 'installment_USD', 'funded_amnt_EUR', 'total_pymnt_USD', 'installment_EUR', 'exchange_rate'],
          dtype='<U19')




```python
arr_numeric[:,5]
```




    array([13.33, 28.99, 28.99, ..., 28.99, 16.55, 28.99])




```python
# Vamos apenas dividir por 100
arr_numeric[:,5] = arr_numeric[:,5] / 100
```


```python
arr_numeric[:,5]
```




    array([0.13, 0.29, 0.29, ..., 0.29, 0.17, 0.29])



## Checkpoint com Variáveis Numéricas Limpas e Pré-Processadas

**Checkpoint 3**


```python
checkpoint_numeric = checkpoint("dados/Checkpoint-Numeric", header_numeric, arr_numeric)
```


```python
checkpoint_numeric['header'], checkpoint_numeric['data']
```




    (array(['id', 'loan_amnt_USD', 'exchange_rate', 'funded_amnt_USD', 'loan_amnt_EUR', 'int_rate', 'installment_USD', 'funded_amnt_EUR', 'total_pymnt_USD', 'installment_EUR', 'exchange_rate'],
           dtype='<U19'),
     array([[48010226.  ,    35000.  ,    31933.3 , ...,     9452.96,     8624.69,        1.1 ],
            [57693261.  ,    30000.  ,    27132.46, ...,     4679.7 ,     4232.39,        1.11],
            [59432726.  ,    15000.  ,    13326.3 , ...,     1969.83,     1750.04,        1.13],
            ...,
            [50415990.  ,    10000.  ,     8910.3 , ...,     2185.64,     1947.47,        1.12],
            [46154151.  ,    35000.  ,    31490.9 , ...,     3199.4 ,     2878.63,        1.11],
            [66055249.  ,    10000.  ,     9145.8 , ...,      301.9 ,      276.11,        1.09]]))



## Construindo o Dataset Final


```python
checkpoint_strings['data'].shape
```




    (10000, 6)




```python
checkpoint_numeric['data'].shape
```




    (10000, 11)




```python
# Concatena os arrays
df_final = np.hstack((checkpoint_numeric['data'], checkpoint_strings['data']))
```


```python
df_final
```




    array([[48010226.  ,    35000.  ,    31933.3 , ...,       13.  ,        1.  ,        1.  ],
           [57693261.  ,    30000.  ,    27132.46, ...,        5.  ,        1.  ,        4.  ],
           [59432726.  ,    15000.  ,    13326.3 , ...,       10.  ,        1.  ,        4.  ],
           ...,
           [50415990.  ,    10000.  ,     8910.3 , ...,        5.  ,        1.  ,        1.  ],
           [46154151.  ,    35000.  ,    31490.9 , ...,       17.  ,        1.  ,        3.  ],
           [66055249.  ,    10000.  ,     9145.8 , ...,        4.  ,        1.  ,        3.  ]])




```python
# Verifica se tem valor ausente
np.isnan(df_final).sum()
```




    0




```python
# Concatena os arrays de nomes de colunas
header_full = np.concatenate((checkpoint_numeric['header'], checkpoint_strings['header']))
```


```python
# Ordenando o dataset
df_final = df_final[np.argsort(df_final[:,0])]
```


```python
df_final
```




    array([[  373332.  ,     9950.  ,     9038.08, ...,       21.  ,        1.  ,        1.  ],
           [  575239.  ,    12000.  ,    10900.2 , ...,       25.  ,        1.  ,        2.  ],
           [  707689.  ,    10000.  ,     8924.3 , ...,       13.  ,        1.  ,        0.  ],
           ...,
           [68614880.  ,     5600.  ,     5121.65, ...,        8.  ,        1.  ,        1.  ],
           [68615915.  ,     4000.  ,     3658.32, ...,       10.  ,        1.  ,        2.  ],
           [68616519.  ,    21600.  ,    19754.93, ...,        3.  ,        1.  ,        2.  ]])




```python
# Conferindo a ordenação da coluna 0
np.argsort(df_final[:,0])
```




    array([   0,    1,    2, ..., 9997, 9998, 9999], dtype=int64)



## Gravando o Dataset Final Limpo e Pré-Processado


```python
# Concatena o array de nomes de colunas com o array de dados
df_final = np.vstack((header_full, df_final))
```


```python
# Salva em disco
np.savetxt("dados/dataset_limpo_preprocessado.csv", 
           df_final, 
           fmt = '%s',
           delimiter = ',')
```

# Fim
