import numpy as np
import theano
import theano.tensor as T
from theano import printing
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from sklearn.neural_network import MLPClassifier
import progressbar

import pickle
import random
from collections import OrderedDict

class VRAE:
    """This class implements the Variational Recurrent Auto Encoder"""
    def __init__(self, hidden_units_encoder, hidden_units_decoder, features, latent_variables, num_drivers, b1=0.9, b2=0.999, learning_rate=0.001, sigma_init=None, batch_size=256, lamda1 = 0.33, lamda_l2=1.0, lamda_l1=1.0, dropout=0.0):
        self.batch_size = batch_size
        self.hidden_units_encoder = hidden_units_encoder
        self.hidden_units_decoder = hidden_units_decoder
        self.features = features
        self.latent_variables = latent_variables
        self.lamda1 = lamda1
        self.lamda_l2 = lamda_l2
        self.lamda_l1 = lamda_l1
        self.keep_prob = np.array(1-dropout).astype(theano.config.floatX)
        self.srng = RandomStreams(np.random.RandomState().randint(999999))

        self.b1 = theano.shared(np.array(b1).astype(theano.config.floatX), name = "b1")
        self.b2 = theano.shared(np.array(b2).astype(theano.config.floatX), name = "b2")
        self.learning_rate = theano.shared(np.array(learning_rate).astype(theano.config.floatX), name="learning_rate")


        #Initialize all variables as shared variables so model can be run on GPU

        #gru encoder
        
        sigma_init = np.sqrt(2.0/(hidden_units_encoder+features))
        U_z = theano.shared(np.random.normal(0,sigma_init,(features,hidden_units_encoder)).astype(theano.config.floatX), name='U_z')
        sigma_init = np.sqrt(1.0/(hidden_units_encoder))
        W_z = theano.shared(np.random.normal(0,sigma_init,(hidden_units_encoder,hidden_units_encoder)).astype(theano.config.floatX), name='W_z')
        b_z = theano.shared(np.zeros((hidden_units_encoder,1)).astype(theano.config.floatX), name='b_z', broadcastable=(False,True))

        sigma_init = np.sqrt(2.0/(hidden_units_encoder+features))
        U_r = theano.shared(np.random.normal(0,sigma_init,(features,hidden_units_encoder)).astype(theano.config.floatX), name='U_r')
        sigma_init = np.sqrt(1.0/(hidden_units_encoder))
        W_r = theano.shared(np.random.normal(0,sigma_init,(hidden_units_encoder,hidden_units_encoder)).astype(theano.config.floatX), name='W_r')
        b_r = theano.shared(np.zeros((hidden_units_encoder,1)).astype(theano.config.floatX), name='b_r', broadcastable=(False,True))

        sigma_init = np.sqrt(2.0/(hidden_units_encoder+features))
        U_h = theano.shared(np.random.normal(0,sigma_init,(features,hidden_units_encoder)).astype(theano.config.floatX), name='U_h')
        sigma_init = np.sqrt(1.0/(hidden_units_encoder))
        W_h = theano.shared(np.random.normal(0,sigma_init,(hidden_units_encoder,hidden_units_encoder)).astype(theano.config.floatX), name='W_h')
        b_h = theano.shared(np.zeros((hidden_units_encoder,1)).astype(theano.config.floatX), name='b_h', broadcastable=(False,True))


        #latent

        sigma_init = np.sqrt(2.0/(latent_variables+hidden_units_encoder))
        W_hmu = theano.shared(np.random.normal(0,sigma_init,(hidden_units_encoder, latent_variables)).astype(theano.config.floatX), name='W_hmu')
        b_hmu = theano.shared(np.zeros((latent_variables,1)).astype(theano.config.floatX), name='b_hmu', broadcastable=(False,True))

        W_hsigma = theano.shared(np.random.normal(0,sigma_init,(hidden_units_encoder, latent_variables)).astype(theano.config.floatX), name='W_hsigma')
        b_hsigma = theano.shared(np.zeros((latent_variables,1)).astype(theano.config.floatX), name='b_hsigma', broadcastable=(False,True))

        # gru decoder
        sigma_init = np.sqrt(2.0/(hidden_units_decoder+features))
        U_dec_z = theano.shared(np.random.normal(0,sigma_init,(features,hidden_units_decoder)).astype(theano.config.floatX), name='U_dec_z')
        sigma_init = np.sqrt(1.0/(hidden_units_decoder))
        W_dec_z = theano.shared(np.random.normal(0,sigma_init,(hidden_units_decoder,hidden_units_decoder)).astype(theano.config.floatX), name='W_dec_z')
        b_dec_z = theano.shared(np.zeros((hidden_units_decoder,1)).astype(theano.config.floatX), name='b_dec_z', broadcastable=(False,True))

        sigma_init = np.sqrt(2.0/(hidden_units_decoder+features))
        U_dec_r = theano.shared(np.random.normal(0,sigma_init,(features,hidden_units_decoder)).astype(theano.config.floatX), name='U_dec_r')
        sigma_init = np.sqrt(1.0/(hidden_units_decoder))
        W_dec_r = theano.shared(np.random.normal(0,sigma_init,(hidden_units_decoder,hidden_units_decoder)).astype(theano.config.floatX), name='W_dec_r')
        b_dec_r = theano.shared(np.zeros((hidden_units_decoder,1)).astype(theano.config.floatX), name='b_dec_r', broadcastable=(False,True))

        sigma_init = np.sqrt(2.0/(hidden_units_decoder+features))
        U_dec_h = theano.shared(np.random.normal(0,sigma_init,(features,hidden_units_decoder)).astype(theano.config.floatX), name='U_dec_h')
        sigma_init = np.sqrt(1.0/(hidden_units_decoder))
        W_dec_h = theano.shared(np.random.normal(0,sigma_init,(hidden_units_decoder,hidden_units_decoder)).astype(theano.config.floatX), name='W_dec_h')
        b_dec_h = theano.shared(np.zeros((hidden_units_decoder,1)).astype(theano.config.floatX), name='b_dec_h', broadcastable=(False,True))

        sigma_init = np.sqrt(2.0/(hidden_units_decoder+features))
        W_hx = theano.shared(np.random.normal(0,sigma_init,(hidden_units_decoder,features)).astype(theano.config.floatX), name='W_hx')
        b_hx = theano.shared(np.zeros((features,1)).astype(theano.config.floatX), name='b_hx', broadcastable=(False,True))

        #decoder
        W_zh = theano.shared(np.random.normal(0,sigma_init,(latent_variables, hidden_units_decoder)).astype(theano.config.floatX), name='W_zh')
        b_zh = theano.shared(np.zeros((hidden_units_decoder,1)).astype(theano.config.floatX), name='b_zh', broadcastable=(False,True))


        #driver
        sigma_init = np.sqrt(2.0/(hidden_units_encoder + num_drivers))
        W_driver = theano.shared(np.random.normal(0,sigma_init,(hidden_units_encoder, num_drivers)).astype(theano.config.floatX), name='W_driver')
        b_driver = theano.shared(np.zeros((num_drivers,1)).astype(theano.config.floatX), name='b_driver', broadcastable=(False,True))


        self.params = OrderedDict([("W_z", W_z), ("U_z", U_z), ("b_z", b_z), ("W_r", W_r), ("U_r", U_r), ("b_r", b_r), ("W_h", W_h), ("U_h", U_h), ("b_h", b_h),\
            ("W_dec_z", W_dec_z), ("U_dec_z", U_dec_z), ("b_dec_z", b_dec_z), ("W_dec_r", W_dec_r), ("U_dec_r", U_dec_r), ("b_dec_r", b_dec_r), ("W_dec_h", W_dec_h), ("U_dec_h", U_dec_h), ("b_dec_h", b_dec_h), \
            ("W_hmu", W_hmu), ("b_hmu", b_hmu), \
            ("W_hsigma", W_hsigma), ("b_hsigma", b_hsigma), ("W_zh", W_zh), ("b_zh", b_zh),
            ("W_hx", W_hx), ("b_hx", b_hx), ("W_driver", W_driver), ("b_driver", b_driver),
            ])

        #Adam parameters
        self.m = OrderedDict()
        self.v = OrderedDict()

        for key,value in self.params.items():
            if 'b' in key:
                self.m[key] = theano.shared(np.zeros_like(value.get_value()).astype(theano.config.floatX), name='m_' + key, broadcastable=(False, True))
                self.v[key] = theano.shared(np.zeros_like(value.get_value()).astype(theano.config.floatX), name='v_' + key, broadcastable=(False, True))
            else:
                self.m[key] = theano.shared(np.zeros_like(value.get_value()).astype(theano.config.floatX), name='m_' + key)
                self.v[key] = theano.shared(np.zeros_like(value.get_value()).astype(theano.config.floatX), name='v_' + key)


    def create_gradientfunctions(self, train_data, train_labels, val_data, val_labels):
        """This function takes as input the whole dataset and creates the entire model"""
        def encodingstep(x_t, h_t):
            z_t = T.nnet.sigmoid(T.dot(x_t, self.params['U_z']) + T.dot(h_t, self.params['W_z']).squeeze() + self.params['b_z'].squeeze())
            r_t = T.nnet.sigmoid(T.dot(x_t, self.params['U_r']) + T.dot(h_t, self.params['W_r']).squeeze() + self.params['b_r'].squeeze())
            h = T.tanh(T.dot(x_t, self.params['U_h'])+T.dot(h_t*r_t, self.params['W_h']) + self.params['b_h'].squeeze())
            new_h_t = (1-z_t)*h+z_t*h_t
            return new_h_t

        x = T.tensor3("x")

        h0_enc = T.matrix("h0_enc")
        result, _ = theano.scan(encodingstep, 
                sequences = x, 
                outputs_info = h0_enc)

        h_encoder = result[-1]

        #log sigma encoder is squared
        mu_encoder = T.dot(h_encoder, self.params["W_hmu"]) + self.params["b_hmu"].squeeze()
        log_sigma_encoder = T.dot(h_encoder, self.params["W_hsigma"]) + self.params["b_hsigma"].squeeze()

        #Use a very wide prior to make it possible to learn something with Z
        #logpz = 0.005 * T.sum(1 + log_sigma_encoder - mu_encoder**2 - T.exp(log_sigma_encoder), axis = 1)
        logpz = 0.5 * T.sum(1 + log_sigma_encoder - mu_encoder**2 - T.exp(log_sigma_encoder), axis = 1)

        if "gpu" in theano.config.device:
            srng = theano.sandbox.cuda.rng_curand.CURAND_RandomStreams()
        else:
            srng = T.shared_randomstreams.RandomStreams()

        #Reparametrize Z
        eps = srng.normal((x.shape[1], self.latent_variables), avg = 0.0, std = 1.0, dtype=theano.config.floatX)
        z = mu_encoder + T.exp(0.5 * log_sigma_encoder) * eps

        h0_dec = T.tanh(T.dot(z, self.params["W_zh"]) + self.params["b_zh"].squeeze())

        def decodingstep(x_t, h_t):
            z_dec_t = T.nnet.sigmoid(T.dot(x_t, self.params['U_dec_z']) + T.dot(h_t, self.params['W_dec_z']) + self.params['b_dec_z'].squeeze())
            r_dec_t = T.nnet.sigmoid(T.dot(x_t, self.params['U_dec_r']) + T.dot(h_t, self.params['W_dec_r']) + self.params['b_dec_r'].squeeze())
            h = T.tanh(T.dot(x_t, self.params['U_dec_h'])+T.dot(h_t*r_dec_t, self.params['W_dec_h']) + self.params['b_dec_h'].squeeze())
            new_h_t = (1-z_dec_t)*h + z_dec_t*h_t
            new_x_t = T.tanh(h.dot(self.params["W_hx"]) + self.params["b_hx"].squeeze())
            return new_x_t, new_h_t


        x0 = T.matrix("x0")
        [y, _], _ = theano.scan(decodingstep,
                n_steps = x.shape[0], 
                outputs_info = [x0, h0_dec])

        # Clip y to avoid NaNs, necessary when lowerbound goes to 0
        # 128 x 8 x 35
        y = T.clip(y, -1 + 1e-6, 1 - 1e-6)
        logpxz = -T.sum(T.pow(y-x, 2), axis = 0)
        logpxz = T.mean(logpxz, axis = 1)

        #Average over batch dimension
        logpx = T.mean(logpxz + logpz)

        #Driver output
        batch_start = T.iscalar('batch_start')
        batch_end = T.iscalar('batch_end')
        labels = T.ivector('labels')
        train_labels = theano.shared(train_labels.astype('int32'))
        val_labels = theano.shared(val_labels.astype('int32'))
        keep_prob = T.scalar(dtype=theano.config.floatX)


        mask = self.srng.binomial(p=keep_prob, size=(self.hidden_units_encoder,)).astype(theano.config.floatX)/keep_prob
        printer = printing.Print('')

        driver_output = T.nnet.softmax(T.dot(h_encoder*mask, self.params['W_driver']) + self.params['b_driver'].squeeze())

        max_minus_min = (driver_output.max(axis=0)-driver_output.min(axis=0)).sum()
        var = (driver_output.var(axis=0)).sum()
        mean = (driver_output.mean(axis=0)).sum()

        cross_entropy = T.nnet.categorical_crossentropy(driver_output, labels)


        driver_loss = (-T.mean(cross_entropy))
        l1_loss = (-T.sum([T.sum(abs(v)) for v in self.params.values()]))
        l2_loss = (-T.sum([T.sum(v**2) for v in self.params.values()]))

        #Compute all the gradients
        total_loss = ((1-self.lamda1) * logpx + self.lamda1*driver_loss + self.lamda_l2*l2_loss + self.lamda_l1*l1_loss)
        gradients = T.grad(total_loss, self.params.values(), disconnected_inputs='ignore')

        #Let Theano handle the updates on parameters for speed
        updates = OrderedDict()
        epoch = T.iscalar("epoch")
        gamma = (T.sqrt(1 - (1 - self.b2)**epoch)/(1 - (1 - self.b1)**epoch)).astype(theano.config.floatX)

        #Adam
        for parameter, gradient, m, v in zip(self.params.values(), gradients, self.m.values(), self.v.values()):
            new_m = self.b1 * gradient + (1 - self.b1) * m
            new_v = self.b2 * (gradient**2) + (1 - self.b2) * v

            updates[parameter] = parameter + self.learning_rate * gamma * new_m / (T.sqrt(new_v)+ 1e-8)
            updates[m] = new_m
            updates[v] = new_v

        train_data = theano.shared(train_data.transpose(1,0,2)).astype(theano.config.floatX)

        givens = {
            h0_enc: T.zeros((batch_end-batch_start, self.hidden_units_encoder)).astype(theano.config.floatX), 
            x0:     T.zeros((batch_end-batch_start, self.features)).astype(theano.config.floatX),
            x:      train_data[:,batch_start:batch_end,:],
            labels: train_labels[batch_start:batch_end],
            keep_prob: self.keep_prob
        }

        self.updatefunction = theano.function([epoch, batch_start, batch_end], [logpxz.mean(), driver_loss], updates=updates, givens=givens, allow_input_downcast=True)

        x_val = theano.shared(val_data.transpose(1, 0, 2)).astype(theano.config.floatX)
        givens[x] = x_val[:, batch_start:batch_end,:]
        givens[labels] = val_labels[batch_start:batch_end]
        givens[keep_prob] = np.array(1.0).astype(theano.config.floatX)
        self.likelihood = theano.function([batch_start, batch_end], [logpxz.mean(), driver_loss, max_minus_min, var, mean], givens=givens)

        x_test = T.tensor3("x_test")
        test_givens = {
            x: x_test,
            h0_enc: T.zeros((x_test.shape[1], self.hidden_units_encoder)).astype(theano.config.floatX), 
        }

        self.encoder = theano.function([x_test], h_encoder, givens=test_givens)
        h_e = T.matrix('h_e')
        self.driver_predict = theano.function([h_e], driver_output, givens={h_encoder: h_e, keep_prob: np.array(1.0).astype(theano.config.floatX)})


        return True


    def save_parameters(self, path):
        """Saves all the parameters in a way they can be retrieved later"""
        pickle.dump({name: p.get_value() for name, p in self.params.items()}, open(path + "/params.pkl", "wb"))
        pickle.dump({name: m.get_value() for name, m in self.m.items()}, open(path + "/m.pkl", "wb"))
        pickle.dump({name: v.get_value() for name, v in self.v.items()}, open(path + "/v.pkl", "wb"))

    def load_parameters(self, path):
        """Load the variables in a shared variable safe way"""
        p_list = pickle.load(open(path + "/params.pkl", "rb"))
        m_list = pickle.load(open(path + "/m.pkl", "rb"))
        v_list = pickle.load(open(path + "/v.pkl", "rb"))

        for name in p_list.keys():
            self.params[name].set_value(p_list[name].astype(theano.config.floatX))
            self.m[name].set_value(m_list[name].astype(theano.config.floatX))
            self.v[name].set_value(v_list[name].astype(theano.config.floatX))

def copy_the_model(original_model, new_model):
    for (key, value) in original_model.params.items():
        new_model.params[key].set_value(value.get_value())

def numpy_ce_loss(model_out, expected_out):
    if model_out.shape[1] == 1:
        return -(np.sum(np.log(model_out) * expected_out + (np.log(1-model_out))*(1-expected_out))/model_out.shape[0])
    else:
        return -(np.sum(np.log(model_out) * expected_out)/model_out.shape[0])

def shuffle_in_union(data, keys):
    rng_state = np.random.get_state()
    np.random.shuffle(keys)
    np.random.set_state(rng_state)
    np.random.shuffle(data)
    np.random.set_state(rng_state)
    return data, keys

def returnTrainAndTestData():

    matrices = np.load('./data/RandomSample_5_10.npy')
    keys = pickle.load(open('./data/RandomSample_5_10.pkl'))
    FEATURES = matrices.shape[-1]
    
    train_data = []
    train_labels = []
    dev_data = []
    dev_labels = []
    test_data = []
    test_labels = []
    
    curTraj = ''
    assign = ''
    test_traj = []
    traj_to_driver = {}
    
    driverIds = {}
    
    for idx in range(len(keys)):
        d,t = keys[idx]
        if d in driverIds:
            dr = driverIds[d]
        else: 
            dr = len(driverIds)
            driverIds[d] = dr
        m = matrices[idx][1:129,]
        if t != curTraj:
            curTraj = t
            r = random.random()                    
        if r < 0.75:
          train_data.append(m)
          train_labels.append(dr)
        elif r < 0.85:
            dev_data.append(m)
            dev_labels.append(dr)
        else:
          test_traj.append(t)
          test_data.append(m)
          test_labels.append(dr)        
          traj_to_driver[t] = dr

    train_data   = np.asarray(train_data, dtype="float32")
    train_labels = np.asarray(train_labels, dtype="int32")
    dev_data     = np.asarray(dev_data, dtype="float32")
    dev_labels   = np.asarray(dev_labels, dtype="int32")
    test_data    = np.asarray(test_data, dtype="float32")
    test_labels  = np.asarray(test_labels, dtype="int32")

    train_data, train_labels = shuffle_in_union(train_data, train_labels)
    
    if len(dev_data) == 0:
        train_data, dev_data, train_labels, dev_labels = train_test_split(train_data, train_labels, test_size=0.2)
    return train_data, train_labels, dev_data, dev_labels, test_data, test_labels, test_traj, traj_to_driver, FEATURES

if __name__ == "__main__":
    MAX_EARLY_STOP_COUNT = 5
    rnn_size = 256
    latent_size = 256
    num_drivers = 50
    batch_size = 256
    lamda1 = 1/3
    lamda2 = 1/3
    lamda_l1 = 0.0
    lamda_l2 = 0.0
    dropout = 0
    num_epochs = 200


    x_train, t_train, x_valid, t_valid, x_test, t_test, test_traj, traj_to_driver, num_features = returnTrainAndTestData()
    
    model = VRAE(rnn_size, rnn_size, num_features, latent_size, num_drivers, batch_size=batch_size, lamda1=lamda1, lamda_l2=lamda_l2, lamda_l1=lamda_l1, dropout=dropout)
    saved_model = VRAE(rnn_size, rnn_size, num_features, latent_size, num_drivers, batch_size=batch_size, lamda1=lamda1, lamda_l2=lamda_l2, lamda_l1=lamda_l1)


    batch_order = np.arange(x_train.shape[0] // model.batch_size + 1)
    val_batch_order = np.arange(x_valid.shape[0] // model.batch_size + 1)
    epoch = 0
    LB_list = []

    model.create_gradientfunctions(x_train, t_train, x_valid, t_valid)
    saved_model.create_gradientfunctions(x_train, t_train, x_valid, t_valid)

    print("iterating")
    best_val_score = -float("inf")
    prev_val_score = -float("inf")
    early_stop_count = MAX_EARLY_STOP_COUNT

    while epoch <  num_epochs and early_stop_count > 0:
        epoch += 1
        np.random.shuffle(batch_order)
        train_total_loss = 0.0
        train_driver_loss = 0.0
        val_total_loss = 0.0
        val_driver_loss = 0.0

        bar = progressbar.ProgressBar()

        for batch in bar(batch_order):
            batch_end = min(model.batch_size*(batch+1), x_train.shape[0])
            batch_start = model.batch_size*batch
            l1, l2 = model.updatefunction(epoch, batch_start, batch_end)
            train_total_loss += (l1+l2)*(batch_end-batch_start)
            train_driver_loss += l2*(batch_end-batch_start)

        train_total_loss /= x_train.shape[0]
        train_driver_loss /= x_train.shape[0]

        print("Epoch {0} finished. Total Training Loss: {1}, Driver Loss: {2}".format(epoch, train_total_loss, train_driver_loss))

        bar = progressbar.ProgressBar()
        valid_LB = 0.0
        val_LB1 = 0.0
        val_LB2 = 0.0
        S_mean  = 0.0
        S_mmm = 0.0
        S_var = 0.0

        for batch in bar(val_batch_order):
            batch_end = min(model.batch_size*(batch+1), x_valid.shape[0])
            batch_start = model.batch_size*batch
            l1, l2, s_mmm, s_var, s_mean = model.likelihood(batch_start, batch_end)
            val_total_loss += (l1+l2)*(batch_end-batch_start)
            val_driver_loss += l2*(batch_end-batch_start)


        val_total_loss /= x_valid.shape[0]
        val_driver_loss /= x_valid.shape[0]

        print("Val loss: {}, Val driver loss: {}".format(val_total_loss, val_driver_loss))

        ###Classification
        h_train = []
        h_val = []

        for i in range(x_train.shape[0]//model.batch_size+1):
            h_train.append(model.encoder(x_train[i*model.batch_size:(i+1)*model.batch_size].transpose(1, 0, 2).astype(theano.config.floatX)))
        for i in range(x_valid.shape[0]//model.batch_size+1):
            h_val.append(model.encoder(x_valid[i*model.batch_size:(i+1)*model.batch_size].transpose(1, 0, 2).astype(theano.config.floatX)))

        h_train = np.concatenate(h_train)
        h_val = np.concatenate(h_val)

        clf = MLPClassifier(hidden_layer_sizes=(), verbose=0, max_iter=70, batch_size=256)
        clf.fit(h_train, t_train)

        train_predictions = clf.predict(h_train)
        val_predictions = clf.predict(h_val)

        train_score = np.mean(train_predictions == t_train)
        val_score = np.mean(val_predictions == t_valid)

        print("Accuracy on train: %0.4f" % train_score)
        print("Accuracy on val: %0.4f" % val_score)

        if val_score > best_val_score:
            print "Updating model"
            best_val_score = val_score
            copy_the_model(model, saved_model)

        if val_score < prev_val_score:
            early_stop_count -= 1
            print("Early stopping count reduced to " + str(early_stop_count))
        else:
            early_stop_count = MAX_EARLY_STOP_COUNT

        prev_val_score = val_score

    h_test = []
    for i in range(x_test.shape[0]//saved_model.batch_size+1):
        h_test.append(saved_model.encoder(x_test[i*saved_model.batch_size:(i+1)*saved_model.batch_size].transpose(1, 0, 2).astype(theano.config.floatX)))
    h_test = np.concatenate(h_test)
    print("Accuracy on test: %0.4f" % clf.get_accuracy(h_test, Y_test))

    y_test = clf.predict(h_test)
    df = pd.DataFrame(y_test)
    df['traj'] = test_traj
    groupby = df.groupby('traj').sum()
    trip_preds = groupby.idxmax(1)
    trip_accuracy = float(sum([traj_to_driver[k]==v for (k, v) in trip_preds.iteritems()]))/trip_preds.shape[0]
    print("Trip level prediction accuracy = {:.2f}".format(trip_accuracy))
