import copy
import random
from Client_COVID import Client
import traceback


class Server:
    def __init__(self, num_clients=5):
        self.num_clients = num_clients
        self.clients = [Client() for _ in range(num_clients)]
        self.global_weights_d = None
        self.global_weights_g = None

    def FedAvg(self, weights):
        """ Federated Averaging """
        w_avg = copy.deepcopy(weights[0])
        for key in w_avg.keys():
            for i in range(1, len(weights)):
                w_avg[key] += weights[i][key]
            w_avg[key] = w_avg[key] / len(weights)
        return w_avg

    def run(self):
        try:
            selected_clients = random.sample(self.clients, self.num_clients)
            weights_d, weights_g, losses_d, losses_g = self.client_training(selected_clients)
            self.global_weights_d = self.FedAvg(weights_d)
            self.global_weights_g = self.FedAvg(weights_g)
            for client in selected_clients:
                client.client_update(self.global_weights_d, self.global_weights_g)
        except Exception as e:
            print(f"Error during training: {e}")
            traceback.print_exc()

    def client_training(self, selected_clients):
        weights_d, weights_g, losses_d, losses_g = [], [], [], []
        for client in selected_clients:
            w_d, w_g, loss_d, loss_g = client.local_training()
            weights_d.append(w_d)
            weights_g.append(w_g)
            losses_d.append(loss_d)
            losses_g.append(loss_g)
        return weights_d, weights_g, losses_d, losses_g


if __name__ == "__main__":
    server = Server(num_clients=5)
    server.run()
