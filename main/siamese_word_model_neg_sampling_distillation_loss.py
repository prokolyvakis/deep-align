import theano
import numpy as np
from theano import tensor as T
from theano import config
from theano.ifelse import ifelse
from lasagne_average_layer import lasagne_average_layer
import lasagne

class softMaxLayer(lasagne.layers.Layer):
    def __init__(self, incoming, **kwargs):
        super(softMaxLayer, self).__init__(incoming, **kwargs)

    def get_output_shape_for(self, input_shape):
        '''
        The input is just a vector of numbers.
        The output is also a vector, same size as the input.
        '''
        return input_shape

    def get_output_for(self, input, **kwargs):
        '''
        Take the exp() of all inputs, and divide by the total.
        '''

        inputs = input
        eps = 1e-9

        exps = T.exp( T.sqrt(T.maximum(inputs, eps)) )

        return exps / exps.sum(axis=1).dimshuffle((0, 'x'))

class cosineLayer(lasagne.layers.MergeLayer):
    '''
    First layer gives a vector (or a batch of vectors, really)
    Second layer gives a matrix (well, a batch of matrices)
    We return a vector of numbers, just as many as there are cols in the second 
    layer matrix (NOTE that the second input layer is a transposed version of
    the layer before it)
    '''

    def __init__(self, incomings, iEmbeddingSize, **kwargs):
        super(cosineLayer, self).__init__(incomings, **kwargs)

        self.iEmbeddingSize = iEmbeddingSize

    def get_output_shape_for(self, input_shapes):
        # input_shapes come like this:
        #  [(batch_size, vectors_size), (batch_size, rows, cols)]
        return (input_shapes[0][0], input_shapes[1][1])

    def get_output_for(self, inputs, **kwargs):
        '''
        We want a dot product of every row in inputs[0] (a vector) with every
        row in inputs[1] (a matrix).
        We do this 'by hand': we do a element-wise multiplication of every vector
        in inputs[0] with every matrix in inputs[1], and sum the result.
        '''
        dots = (inputs[0].reshape((-1, self.iEmbeddingSize, 1)) * \
                  inputs[1]).sum(axis=1)

        # Make sure the braodcasting is right
        norms_1 = T.sqrt(T.square(inputs[0]).sum(axis=1)).dimshuffle(0, 'x')
        # NOTE that the embeddings are transposed in the previous layer 
        norms_2 = T.sqrt(T.square(inputs[1]).sum(axis=1))

        norms = norms_1 * norms_2

        return dots / norms

class averageLayer_matrix(lasagne.layers.MergeLayer):
    def __init__(self, incomings, **kwargs):
        super(averageLayer_matrix, self).__init__(incomings, **kwargs)


    def get_output_shape_for(self, input_shape):
        '''
        The input is a batch of matrices of word vectors.
        The output is a batch of vectors, one for each matrix, the same size as
        the input word embeddings
        In other words, since we are averaging, we loose the penultimate dimension
        '''
        return (input_shape[0][0], input_shape[0][1], input_shape[0][3])

    def get_output_for(self, inputs, **kwargs):
        '''
        The input is a batch of matrices of word vectors.
        The output the sum of the word embeddings divided by the number of
        non-zero word embeddings in the input.
        '''

        emb = inputs[0]
        mask = inputs[1]
        emb = (emb * mask[:, :, :, None]).sum(axis=1)
        averages = emb / mask.sum(axis=1)[:, :, None]

        return averages

class ppdb_word_model(object):

    def __init__(self, We_initial, params):

        params.siamese = True
        ## Params
        initial_We = theano.shared(np.asarray(We_initial, dtype = config.floatX))
        We = theano.shared(np.asarray(We_initial, dtype = config.floatX))

        ## Symbolic Params
        # Input variable for a batch of sentences who seek 
        # for target synonyms and antonyms in the next tensor
        senBatch_indices = T.imatrix(); senMask = T.matrix()

        # Input variable for a batch of positive and negative
        # examples (so syn, neg1, neg2, ...)
        targetBatch_indices = T.itensor3(); targetMask = T.tensor3()
        
        targets = T.matrix()
        

        ## First embedding input layer
        l_in_1   = lasagne.layers.InputLayer((None, None, 1))
        l_mask_1 = lasagne.layers.InputLayer(shape=(None, None))
        # First embedding layer and Knowledge Distillation's embedding Layer
        l_emb_1  = lasagne.layers.EmbeddingLayer(l_in_1, input_size=We.get_value().shape[0], output_size=We.get_value().shape[1], W=We)
        l_emb_1_reg  = lasagne.layers.EmbeddingLayer(l_in_1, input_size=initial_We.get_value().shape[0], output_size=initial_We.get_value().shape[1], W=initial_We)
        l_emb_1_reg.params[l_emb_1_reg.W].remove('trainable')
        # First Average Layer and Knowledge Distillation's First Average Layer
        #l_emb_1_drop = lasagne.layers.DropoutLayer(l_emb_1, p=0.8)
        l_average_1 = lasagne_average_layer([l_emb_1, l_mask_1])
        l_average_1_reg = lasagne_average_layer([l_emb_1_reg, l_mask_1])


        in2embgs = lasagne.layers.get_output(l_emb_1, {l_in_1:senBatch_indices}, deterministic=True)
        embg1 = lasagne.layers.get_output(l_average_1, {l_in_1:senBatch_indices, l_mask_1:senMask}, deterministic=True)



        ## Second embedding input layer
        l_in_2   = lasagne.layers.InputLayer(shape=(None, None, None, 1))
        l_mask_2 = lasagne.layers.InputLayer(shape=(None, None, None))
        # Second embedding layer, the weights tied with the first embedding layer
        l_emb_2  = lasagne.layers.EmbeddingLayer(l_in_2, input_size=We.get_value().shape[0], output_size=We.get_value().shape[1], W=l_emb_1.W)
        l_emb_2_reg  = lasagne.layers.EmbeddingLayer(l_in_2, input_size=initial_We.get_value().shape[0], output_size=initial_We.get_value().shape[1], W=l_emb_1_reg.W)
        l_emb_2_reg.params[l_emb_2_reg.W].remove('trainable')
        


        # Second Average Layer
        #l_emb_2 = lasagne.layers.DropoutLayer(l_emb_2, p=0.8)
        l_average_2 = averageLayer_matrix([l_emb_2, l_mask_2])
        l_transpose_2 = lasagne.layers.DimshuffleLayer(l_average_2, (0,2,1))
        # Knowledge Distillation's Second Average Layer
        l_average_2_reg = averageLayer_matrix([l_emb_2_reg, l_mask_2])
        l_transpose_2_reg = lasagne.layers.DimshuffleLayer(l_average_2_reg, (0,2,1))

        
        ## Layer Combination
        l_cosine = cosineLayer([l_average_1, l_transpose_2], We.get_value().shape[1])
        g1g2 = lasagne.layers.get_output(l_cosine, {l_in_1:senBatch_indices, \
                    l_mask_1:senMask, l_in_2:targetBatch_indices, l_mask_2:targetMask}, deterministic=True)
        g1g2 = g1g2[:, 0]
        # Knowledge Distillation's Layer Combination
        l_cosine_reg = cosineLayer([l_average_1_reg, l_transpose_2_reg], We.get_value().shape[1])
        

        l_final_layer = softMaxLayer(l_cosine)
        # Knowledge Distillation's Layer Combination
        l_final_layer_reg = softMaxLayer(l_cosine_reg)
        
        ## Objective Function
        prediction = lasagne.layers.get_output(l_final_layer, {l_in_1:senBatch_indices, \
                    l_mask_1:senMask, l_in_2:targetBatch_indices, l_mask_2:targetMask})
        # Knowledge Distillation's Prediction
        prediction_reg = lasagne.layers.get_output(l_final_layer_reg, {l_in_1:senBatch_indices, \
                    l_mask_1:senMask, l_in_2:targetBatch_indices, l_mask_2:targetMask})


        self.all_params = lasagne.layers.get_all_params(l_final_layer, trainable=True)

        loss = lasagne.objectives.categorical_crossentropy(prediction, targets)
        # Knowledge Distillation's Loss
        loss_reg = lasagne.objectives.categorical_crossentropy(prediction, prediction_reg)
        
        
        cost = params.LW*loss_reg.mean() + params.hyper_k1*loss.mean() 

        #feedforward
        self.feedforward_function = theano.function([senBatch_indices,senMask], embg1)
        self.cost_function = theano.function([senBatch_indices, senMask, targetBatch_indices, 
                                              targetMask, targets], cost)
        self.cost_distillation = theano.function([senBatch_indices, senMask, targetBatch_indices, 
                                              targetMask], loss_reg.mean())
       
        self.scoring_function = theano.function([senBatch_indices, senMask, 
                                    targetBatch_indices, targetMask], g1g2)


        self.word2embeddings = theano.function([senBatch_indices], in2embgs)


        #updates
        grads = theano.gradient.grad(cost, self.all_params)
        if params.clip:
            grads = [lasagne.updates.norm_constraint(grad, params.clip, range(grad.ndim)) for grad in grads]
        updates = params.learner(grads, self.all_params, params.eta)
        self.train_function = theano.function([senBatch_indices, senMask, 
                                    targetBatch_indices, targetMask, targets], cost, updates=updates)