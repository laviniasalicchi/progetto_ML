from laplotter import LossAccPlotter

plotter = LossAccPlotter()

for epoch in range(100):
    # somehow generate loss and accuracy with your model
    loss_train, acc_train = your_model.train()
    loss_val, acc_val = your_model.validate()

    # plot the last values
    plotter.add_values(epoch,
                       loss_train=loss_train, acc_train=acc_train,
                       loss_val=loss_val, acc_val=acc_val)

# As the plot is non-blocking, we should call plotter.block() at the end, to
# change it to the blocking-mode. Otherwise the program would instantly end
# and thereby close the plot.
plotter.block()