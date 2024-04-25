class TrainingHistory:
    def __init__(self):
        self.train_losses = []
        self.train_accuracies = []

        self.val_losses = []
        self.val_accuracies = []

    def append_train(self, loss, accuracy):
        self.train_losses.append(loss)
        self.train_accuracies.append(accuracy)

    def append_val(self, loss, accuracy):
        self.val_losses.append(loss)
        self.val_accuracies.append(accuracy)