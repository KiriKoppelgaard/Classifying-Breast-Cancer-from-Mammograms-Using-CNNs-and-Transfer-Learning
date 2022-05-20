
#######Cross Validation test 
kfold = KFold(n_splits=5, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1
batch_size = 100
no_epochs =  10
acc_per_fold, loss_per_fold = [], []

# Define the model architecture
model = cnn(input_shape = input_shape)


for train, test in kfold.split(x_train, y_train):
  # Generate a print
  print(f'Training for fold {fold_no} ...')

  # Fit data to model
  history = model.fit(x_train[train], y_train[train],
              batch_size=batch_size,
              epochs=no_epochs,
              verbose=10)

  # Generate generalization metrics
  scores = model.evaluate(x_train[test], y_train[test], verbose=0)
  print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
  acc_per_fold.append(scores[1] * 100)
  loss_per_fold.append(scores[0])

  # Increase fold number
  fold_no = fold_no + 1

#print outcome
print('acc_per_fold', acc_per_fold)
print('loss_per_fold', loss_per_fold)

print('mean accuracy', np.mean(np.array(acc_per_fold)))
print('mean loss', np.mean(np.array(loss_per_fold)))

