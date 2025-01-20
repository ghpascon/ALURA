import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from time import sleep

def get_map_receita(produtos):
    receita_estados = produtos.groupby('Local da compra')[['Preço']].sum()
    receita_estados = produtos.drop_duplicates(subset = 'Local da compra')[['Local da compra', 'lat', 'lon']].merge(receita_estados, left_on = 'Local da compra', right_index = True).sort_values('Preço', ascending = False)

    fig_mapa_receita = px.scatter_geo(receita_estados,
                                   lat = 'lat',
                                   lon = 'lon',
                                   scope = 'south america',
                                   size = 'Preço',
                                   template = 'seaborn',
                                   hover_name = 'Local da compra',
                                   hover_data = {'lat':False,'lon':False},
                                   title = 'Receita por Estado')
    
    fig_receita_estados = px.bar(receita_estados.head(),
                                            x = 'Local da compra',
                                            y = 'Preço',
                                            text_auto = True,
                                            title = 'Top estados')
    return fig_mapa_receita, fig_receita_estados

def get_mensal_receita(produtos):
    produtos['Data da Compra'] = pd.to_datetime(produtos['Data da Compra'], format="%d/%m/%Y")
    receita_mensal = produtos.set_index('Data da Compra').groupby(pd.Grouper(freq='M'))['Preço'].sum().reset_index()
    receita_mensal['Ano'] = receita_mensal['Data da Compra'].dt.year
    receita_mensal['Mes'] = receita_mensal['Data da Compra'].dt.month_name()
    fig_receita_mensal = px.line(receita_mensal,
                                x = 'Mes',
                                y = 'Preço',
                                markers = True,
                                range_y = (0, receita_mensal.max()),
                                color='Ano',
                                line_dash = 'Ano',
                                title = 'Receita mensal'
                                )
    fig_receita_mensal.update_layout(yaxis_title = 'Receita')
    return fig_receita_mensal

def get_category_receita(produtos):
    receita_categorias = produtos.groupby('Categoria do Produto')[['Preço']].sum().sort_values('Preço', ascending=False)

    fig_receita_categorias = px.bar(receita_categorias,
                                    text_auto = True,
                                    title = 'Receita por categoria')
    fig_receita_categorias.update_layout(yaxis_title = 'Receita')
    return fig_receita_categorias



def get_map_vendas(produtos):
    vendas_estados = produtos.groupby('Local da compra')[['Preço']].count()
    vendas_estados = produtos.drop_duplicates(subset = 'Local da compra')[['Local da compra', 'lat', 'lon']].merge(vendas_estados, left_on = 'Local da compra', right_index = True).sort_values('Preço', ascending = False)

    vendas_estados.rename(columns={'Preço': 'Vendas'}, inplace=True)
    fig_mapa_vendas = px.scatter_geo(vendas_estados,
                                   lat = 'lat',
                                   lon = 'lon',
                                   scope = 'south america',
                                   size = 'Vendas',
                                   template = 'seaborn',
                                   hover_name = 'Local da compra',
                                   hover_data = {'lat':False,'lon':False},
                                   title = 'Vendas por Estado')
    
    fig_vendas_estados = px.bar(vendas_estados.head(),
                                            x = 'Local da compra',
                                            y = 'Vendas',
                                            text_auto = True,
                                            title = 'Top estados (vendas)')
    return fig_mapa_vendas, fig_vendas_estados

def get_mensal_vendas(produtos):
    produtos['Data da Compra'] = pd.to_datetime(produtos['Data da Compra'], format="%d/%m/%Y")
    receita_mensal = produtos.set_index('Data da Compra').groupby(pd.Grouper(freq='M'))['Preço'].count().reset_index()
    receita_mensal['Ano'] = receita_mensal['Data da Compra'].dt.year
    receita_mensal['Mes'] = receita_mensal['Data da Compra'].dt.month_name()

    receita_mensal.rename(columns={'Preço': 'Vendas'}, inplace=True)
    fig_receita_mensal = px.line(receita_mensal,
                                x = 'Mes',
                                y = 'Vendas',
                                markers = True,
                                range_y = (0, receita_mensal.max()),
                                color='Ano',
                                line_dash = 'Ano',
                                title = 'Vendas mensal'
                                )
    fig_receita_mensal.update_layout(yaxis_title = 'Receita')
    return fig_receita_mensal

def get_category_vendas(produtos):
    receita_categorias = produtos.groupby('Categoria do Produto')[['Preço']].count().sort_values('Preço', ascending=False)

    fig_receita_categorias = px.bar(receita_categorias,
                                    text_auto = True,
                                    title = 'Vendas por categoria')
    fig_receita_categorias.update_layout(yaxis_title = 'Vendas')
    return fig_receita_categorias

def get_vendedores(produtos, qtd_vendedores):
    vendedores = pd.DataFrame(produtos.groupby('Vendedor')['Preço'].agg(['sum', 'count']))
    fig_receita_vendedores = px.bar(
        vendedores[['sum']].sort_values('sum', ascending=False).head(qtd_vendedores),
        x='sum',
        y=vendedores[['sum']].sort_values('sum', ascending=False).head(qtd_vendedores).index,
        text_auto=True,
        title=f'Top {qtd_vendedores} vendedores (receita)'
        )
    fig_vendas_vendedores = px.bar(
        vendedores[['count']].sort_values('count', ascending=False).head(qtd_vendedores),
        x='count',
        y=vendedores[['count']].sort_values('count', ascending=False).head(qtd_vendedores).index,
        text_auto=True,
        title=f'Top {qtd_vendedores} vendedores (quantidade de vendas)'
    )
    return fig_receita_vendedores, fig_vendas_vendedores

def formata_num(val, casas=2):
        # Gera dinamicamente o formato com o número de casas decimais desejado
        formato = f"{{:,.{casas}f}}"
        return formato.format(val).replace(",", "X").replace(".", ",").replace("X", ".")

def get_products(query_str):
    status_code = 0
    while status_code != 200:
        url = "https://labdados.com/produtos"
        response = requests.get(url, params=query_str)
        status_code = response.status_code
        if status_code!= 200:
            sleep(500)
    dados = pd.DataFrame.from_dict(response.json())
    return dados
    
def streamlit_page():
    st.set_page_config(
        page_title="DASHBOARD VENDAS",
        layout="wide",
        page_icon=":shopping_trolley:",
        initial_sidebar_state="expanded"
        )
    
    #sidebar
    regioes = ['Brasil', 'Centro-Oeste', 'Nordeste', 'Norte', 'Sudeste', 'Sul']
    st.sidebar.title('Filtros')
    regiao = st.sidebar.selectbox('Região', regioes)
    if regiao == 'Brasil':
        regiao = ''

    todos_anos = st.sidebar.checkbox('Dados de todo o perioodo', value=True)
    if todos_anos:
        ano = ''
    else:
        ano = st.sidebar.slider('Ano', 2020, 2023)
    
    #produtos
    query_str = {
        'regiao':regiao.lower(),
        'ano':ano,
    }
    produtos = get_products(query_str)
    filtro_vendedores = st.sidebar.multiselect('Vendedores', produtos['Vendedor'].unique())
    if filtro_vendedores:
        produtos = produtos[produtos['Vendedor'].isin(filtro_vendedores)]

    st.title("DASHBOARD VENDAS")

    col1, col2 = st.columns(2)
    

    #metricas
    col1, col2 = st.columns(2)
    
    col1.metric("Receita Total", formata_num(produtos['Preço'].sum(), 2))
    col2.metric("QTD. Vendas", formata_num(produtos.shape[0], 0))
    # st.dataframe(produtos)

    #tabs
    tab1,tab2,tab3 = st.tabs(['RECEITA', 'VENDAS', 'VENDEDORES'])

    with tab1:
        col1, col2 = st.columns(2)

        #graficos
        fig_mapa_receita, fig_receita_estados = get_map_receita(produtos)
        col1.plotly_chart(fig_mapa_receita, use_container_width=True)
        col2.plotly_chart(fig_receita_estados, use_container_width=True)

        col1.plotly_chart(get_mensal_receita(produtos), use_container_width=True)
        col2.plotly_chart(get_category_receita(produtos))

    with tab2:
        col1, col2 = st.columns(2)

        #graficos
        fig_mapa_vendas, fig_receita_vendas = get_map_vendas(produtos)
        col1.plotly_chart(fig_mapa_vendas, use_container_width=True)
        col2.plotly_chart(fig_receita_vendas, use_container_width=True)

        col1.plotly_chart(get_mensal_vendas(produtos), use_container_width=True)
        col2.plotly_chart(get_category_vendas(produtos))        

    with tab3:
        qtd_vendedores = st.number_input('QTD de vendedores', 2, 10, 5)
        
        receita_vendedores, vendas_vendedores = get_vendedores(produtos, qtd_vendedores)
        
        col1, col2 = st.columns(2)
        
        col1.plotly_chart(receita_vendedores)
        col2.plotly_chart(vendas_vendedores)

if __name__=="__main__":
    streamlit_page()

#streamlit run streamlit/0-vendas.py
