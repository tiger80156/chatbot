from dataPreprocessing import dataPreprocessing, filterTheNonFrequentWord, addToken, sentenceEncoding, sortByLen, split_into_batches
from seq2seqModel import model_input, seq2seq_model
import tensorflow as tf
import time

if __name__== "__main__":
    questions, answers = dataPreprocessing()

    questionsWord2Id, answersWord2Id, Id2Word = filterTheNonFrequentWord(questions, answers, threshold=20)

    answers = addToken(answers)

    questions, answers = sentenceEncoding(questions, answers, questionsWord2Id, answersWord2Id)

    sorted_questions, sorted_answers = sortByLen(questions, answers)

    # Hyperparameter
    EPOCH = 100
    BATCH_SIZE = 64
    RNN_SIZE = 512
    NUM_LAYERS = 3
    ENCODING_EMBEDING_SIZE = 512
    DECODING_EMBEDING_SIZE = 512
    LEARNING_RATE = 0.01
    LEARNING_RATE_DECAY = 0.9
    MIN_LEARNING_RATE = 0.0001
    KEEP_PROBABILITEY = 0.5
    SPLIT_SIZE = 0.15

    # Defining the seccsion
    tf.reset_default_graph()
    session = tf.InteractiveSession()

    # Loading the model input
    inputs, targets, lr, keep_prob = model_input()

    # Setting the seq2seq model length
    sequence_length = tf.placeholder_with_default(25, None, name = "sequence_length")

    # Getting the shape of input tensor
    input_shape = tf.shape(inputs)

    # Getting the training and testing prediction
    training_predictions, testing_prediictions = seq2seq_model(tf.reverse(inputs,[-1]),
                                                                targets,
                                                                keep_prob,
                                                                BATCH_SIZE,
                                                                sequence_length,
                                                                len(answersWord2Id),
                                                                len(questionsWord2Id),
                                                                ENCODING_EMBEDING_SIZE,
                                                                DECODING_EMBEDING_SIZE,
                                                                RNN_SIZE,
                                                                NUM_LAYERS,
                                                                questionsWord2Id)
    with tf.name_scope("optimization"):
        loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions,
                                                        targets,
                                                        tf.ones([input_shape[0], sequence_length]))
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        gradients = optimizer.compute_gradients(loss_error)

        clip_gradients = [(tf.clip_by_value(grad_tensor, -5., 5.),grad_variable)
                            for grad_tensor, grad_variable in gradients if grad_tensor is not None]

        optimizer_gradient_clip = optimizer.apply_gradients(clip_gradients)

    # Seperate the training and validation answer
    training_validation_split = int(len(sorted_questions) * SPLIT_SIZE)
    train_questions = sorted_questions[training_validation_split:]
    train_answers = sorted_answers[training_validation_split:]
    validation_questions = sorted_questions[:training_validation_split]
    validation_answers = sorted_answers[:training_validation_split]


    # Training
    batch_index_check_training_loss = 100
    batch_index_check_validation_loss = ((len(train_questions)) // BATCH_SIZE // 2) -1
    total_training_loss_error =  0
    list_validation_loss_error = []
    early_stopping_check = 0
    early_stopping_stop = 0
    check_point = 'chatbot.weights.ckpt'
    session.run(tf.global_variables_initializer())

    # Train epoches
    for epoch in range(1,EPOCH+1):

        # Seperate the training set into batches
        for batch_index, (padded_questions_in_batches, padded_answers_in_batches) in enumerate(split_into_batches(train_questions, train_answers, BATCH_SIZE, questionsWord2Id, answersWord2Id)):
            start_time = time.time()

            _, batches_training_loss_error = session.run([optimizer_gradient_clip, loss_error],
                                                        {inputs: padded_questions_in_batches,
                                                        targets: padded_answers_in_batches,
                                                        lr: LEARNING_RATE,
                                                        sequence_length: padded_answers_in_batches.shape[1],
                                                        keep_prob: KEEP_PROBABILITEY})

            total_training_loss_error += batches_training_loss_error
            end_time = time.time()
            batch_time = end_time - start_time

            # Print the validation info. at the check point
            if batch_index % batch_index_check_training_loss == 0:
                print("epoch: {:>3}/{}, Batch: {:>4}/{}, Train loss error: {:>6.3f}, Training Time on 100 Batches: {:d} seconds".format(epoch,
                             EPOCH,
                             batch_index,
                             len(train_questions) // BATCH_SIZE,
                             total_training_loss_error / batch_index_check_training_loss,
                             int(batch_time * batch_index_check_training_loss)
                             ))
                total_training_loss_error = 0

            # Print the validation info. at the check point
            if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
                total_validation_loss_error = 0
                start_time = time.time()
                for batch_index_validation, (padded_questions_in_batches, padded_answers_in_batches) in enumerate(split_into_batches(validation_questions, validation_answers, BATCH_SIZE, questionsWord2Id, answersWord2Id)):

                    batches_validation_loss_error = session.run(loss_error,
                                                                {inputs: padded_questions_in_batches,
                                                                targets: padded_answers_in_batches,
                                                                lr: LEARNING_RATE,
                                                                sequence_length: padded_answers_in_batches.shape[1],
                                                                keep_prob: 1 })
                    total_validation_loss_error += batches_validation_loss_error
                end_time = time.time()
                batch_time = end_time - start_time
                average_validation_loss_error = total_validation_loss_error / (len(validation_questions) / BATCH_SIZE)

                print("Validation loss error: {:>6.3f}, Batch validation time: {:d} seconds".format(average_validation_loss_error, int(batch_time)))

                LEARNING_RATE *= LEARNING_RATE_DECAY

                if LEARNING_RATE < MIN_LEARNING_RATE:
                    LEARNING_RATE == MIN_LEARNING_RATE

                list_validation_loss_error.append(average_validation_loss_error)

                if average_validation_loss_error <= min(list_validation_loss_error):
                    print("I get improve now.")
                    early_stopping_check = 0
                    saver = tf.train.Saver()
                    saver.save(session, check_point)

                else:
                    print("I do not speak better. Let's me get more pratice")
                    early_stopping_stop += 1

                    if early_stopping_check == early_stopping_stop:
                        print("My apologies, I can speak better more.")
                        break

print("Training is over")

