from cli import start

if __name__ == "__main__":
    for n in range(1, 6):
        start(data_type="csv", start_params=f"Parameters_{str(n)}.csv", act=0)
