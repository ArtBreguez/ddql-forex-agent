import subprocess
from datetime import datetime, timedelta
import pytz
import json
import pandas as pd

def baixar_dados(par, from_date, to_date):
    try:
        datetime.strptime(from_date, "%Y-%m-%d")
        datetime.strptime(to_date, "%Y-%m-%d")
    except ValueError:
        print("Formato de data inválido. Use o formato YYYY-MM-DD.")
        return

    comando_shell = f"npx dukascopy-node -i {par} -from {from_date} -to {to_date} -t h1 -f json --volumes"

    processo = subprocess.Popen(comando_shell, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    saida, erro = processo.communicate()

    if processo.returncode != 0:
        print(f"Ocorreu um erro: {erro.decode('utf-8')}")
    else:
        print("Comando executado com sucesso.")


if __name__ == "__main__":
    gmt_timezone = pytz.timezone('GMT')
    current_date_gmt = datetime.now(gmt_timezone).date()

    five_days_ago = datetime.now() - timedelta(days=5)

    par = "eurusd"
    from_date = five_days_ago.strftime("%Y-%m-%d")
    to_date = current_date_gmt.strftime("%Y-%m-%d")

    baixar_dados(par, from_date, to_date)

    par_petroleo = "brentcmdusd"
    from_date = five_days_ago.strftime("%Y-%m-%d")
    to_date = current_date_gmt.strftime("%Y-%m-%d")

    baixar_dados(par_petroleo, from_date, to_date)
