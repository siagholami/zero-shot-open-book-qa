# coding=utf-8
# Copyright 2021 - Sia Gholami

from utils import *

logger = gconfig.logger


def compute_loss(y_true, y_pred):

    def compute_position_loss(positions_true, positions_pred):
        ploss = tf.keras.losses.categorical_crossentropy(
            y_true=positions_true, y_pred=positions_pred, from_logits=False, label_smoothing=0)
        return ploss

    yn_loss = tf.keras.losses.categorical_crossentropy(y_true=y_true[0], y_pred=y_pred[0])
    start_loss = compute_position_loss(y_true[1], y_pred[1])
    end_loss = compute_position_loss(y_true[2], y_pred[2])

    return (yn_loss + start_loss + end_loss) / 3.0

#dw
def create_model(
        tfhub_handle_encoder, epochs, training_dataset_cardinality, dropout_rate=0.2):

    encoder_inputs = dict(
        input_word_ids=tf.keras.layers.Input(shape=(None,), dtype=tf.int32),
        input_mask=tf.keras.layers.Input(shape=(None,), dtype=tf.int32),
        input_type_ids=tf.keras.layers.Input(shape=(None,), dtype=tf.int32),
    )

    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')

    outputs = encoder(encoder_inputs)
    XP = outputs['pooled_output']
    XS = outputs['sequence_output']

    # pooled output for yes/no answers
    XP = tf.keras.layers.Dropout(dropout_rate)(XP)
    Yyn_hat = tf.keras.layers.Dense(units=3, activation='softmax', name='classifier')(XP)

    # seq output for text answers
    start_logits = tf.keras.layers.Dense(1, name="start_logit", use_bias=True)(XS)
    start_logits = tf.keras.layers.Flatten()(start_logits)
    Ystart_hat = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(start_logits)

    end_logits = tf.keras.layers.Dense(1, name="end_logit", use_bias=True)(XS)
    end_logits = tf.keras.layers.Flatten()(end_logits)
    Yend_hat = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(end_logits)

    #loss
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

    # optimizer
    num_train_steps = training_dataset_cardinality * epochs
    num_warmup_steps = int(0.1 * num_train_steps)

    init_lr = 3e-5
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')

    model = tf.keras.Model(encoder_inputs, [Yyn_hat, Ystart_hat, Yend_hat])

    model.compile(optimizer=optimizer, loss=[loss, loss, loss], metrics=None)

    return model


def test_model():

    iid = np.array([[  101,  2026,  4578,   102,  1045,  2572,  2416,  2519,  1012,
         2348,  2025,  2590,  2057,  3246,  2017,  1521,  2128,  9107,
         6152, 28727,  6906,  1012,  2115, 10740,  1998, 12247,  2024,
         2054,  1012,  2065,  2019,  2742,  2038,  4242,  3437,  2828,
         2030,  2515,  2025,  5383,  1996,  3437,  1001,  8487,  1010,
         2059,  2057,  2069,  2421,   102],
       [  101,  2026,  4578,   102,  1045,  2572,  2416,  2519,  1012,
         2348,  2025,  2590,  2057,  3246,  2017,  1521,  2128,  9107,
         6152, 28727,  6906,  1012,  2115, 10740,  1998, 12247,  2024,
         2054,  1012,  2065,  2019,  2742,  2038,  4242,  3437,  2828,
         2030,  2515,  2025,  5383,  1996,  3437,  1001,  8487,  1010,
         2059,  2057,  2069,  2421,   102]], dtype='int32')

    sid = np.array([[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1],
       [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1]], dtype="int32")

    mid = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1]], dtype="int32")

    model = create_model(
        tfhub_handle_encoder=gconfig.tfhub_handle_encoder,
        epochs=1,
        training_dataset_cardinality=2,
        dropout_rate=0.2)

    y = model.predict(
        x=dict(
            input_word_ids=iid,
            input_mask=mid,
            input_type_ids=sid
        )
    )
    logger.info(
        f'model out: {y}\n')

    return


def train():

    model = create_model(
        tfhub_handle_encoder=gconfig.tfhub_handle_encoder,
        epochs=1,
        training_dataset_cardinality=2,
        dropout_rate=gconfig.dropout_rate)

    callbacks = [
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{gconfig.models_output_dir}/" + "checkpoint-{epoch:02d}-{val_loss:.4f}.h5",
            monitor=gconfig.earlyStopping_metric,
            verbose=0,
            save_best_only=True,
            save_weights_only=True,
            mode='auto',
            period=1),
        tf.keras.callbacks.EarlyStopping(
            monitor=gconfig.earlyStopping_metric,
            min_delta=gconfig.earlyStopping_min_delta,
            patience=gconfig.earlyStopping_patience,
            verbose=0,
            mode='auto',
            baseline=None,
            restore_best_weights=True)
    ]

    histories = []
    for idx, f in enumerate(gconfig.to_be_trained_files):

        features = load_dict_from_h5(f, limit=gconfig.train_limit)
        X = dict(
            input_word_ids=features['input_word_ids'],
            input_mask=features['input_mask'],
            input_type_ids=features['input_type_ids'])
        Y = [
            features['yes_no_answer'],
            features['text_answer_start_idx'],
            features['text_answer_end_idx']]

        history = model.fit(
            x=X,
            y=Y,
            validation_split=gconfig.validation_split,
            epochs=gconfig.epochs,
            batch_size=gconfig.train_batch,
            callbacks=callbacks,
            verbose=1
        )

        histories.append(history)
        model.save(gconfig.model_dir)

    return histories


def test():
    test_model()
    return None


def main():
    #test()
    train()
    return None


if __name__ == '__main__':
    main()
