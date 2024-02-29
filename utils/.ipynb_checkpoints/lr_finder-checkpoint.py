from keras import backend as K
import math


class ExponentialLearningRate(tf.keras.callbacks.Callback):
    def __init__(self, factor):
        self.factor = factor
        self.rates = []
        self.losses = []

    def on_epoch_begin(self, epoch, logs=None):
        self.sum_of_epoch_losses = 0

    def on_batch_end(self, batch, logs=None):
        mean_epoch_loss = logs["loss"]  # the epoch's mean loss so far 
        new_sum_of_epoch_losses = mean_epoch_loss * (batch + 1)
        batch_loss = new_sum_of_epoch_losses - self.sum_of_epoch_losses
        self.sum_of_epoch_losses = new_sum_of_epoch_losses
        self.rates.append(model.optimizer.variables[1].numpy())
        self.losses.append(batch_loss)
        model.optimizer.variables[1].assign(model.optimizer.variables[1].numpy() * self.factor)
    
        
def find_learning_rate(model, X, epochs=1, batch_size=32, min_rate=1e-6,
                       max_rate=1):
    init_weights = model.get_weights()
    iterations = math.ceil(len(train_paths) / batch_size) * epochs
    factor = (max_rate / min_rate) ** (1 / iterations)
    init_lr = model.optimizer.variables[1].numpy()
    model.optimizer.variables[1].assign(min_rate)
    exp_lr = ExponentialLearningRate(factor)
    history = model.fit(X, epochs=epochs, steps_per_epoch=iterations-1, batch_size=batch_size,
                        callbacks=[exp_lr])
    model.optimizer.variables[1].assign(init_lr)
    return exp_lr.rates, exp_lr.losses

def plot_lr_vs_loss(rates, losses):
    plt.plot(rates, losses, "b")
    plt.gca().set_xscale('log')
    max_loss = losses[0] + min(losses)
    plt.hlines(min(losses), min(rates), max(rates), color="k")
    plt.axis([min(rates), max(rates), 0, max_loss])
    plt.xlabel("Learning rate")
    plt.ylabel("Loss")
    plt.grid()