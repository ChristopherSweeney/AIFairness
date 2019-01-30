import numpy as np
import gensim
import pandas as pd
from matplotlib import pyplot as plt
import seaborn
import re
import statsmodels.formula.api
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
from sklearn.linear_model import LogisticRegression
import re
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

###############################################################################################################################
''''
Utility Methods
'''
###############################################################################################################################
def text_to_vector(embeddings,text):
    tokens = text.split()
    words = filter(lambda x: x in embeddings,[x.lower() for x in words])
    if len(words)>0:
        return np.mean(embeddings[words],axis = 0)
    return None

def words_to_toxicity(model,embeddings,words):
    words = filter(lambda x: x in embeddings,[x.lower() for x in words])
    if len(words)>0:
        vector = np.mean(embeddings[words],axis = 0)
        prob = model.predict_prob(vector)
        return prob
    else: return 0;
    
def text_to_toxicity(model,embeddings,text):
    tokens = text.split()
    toxicity = words_to_toxicity(model,embeddings,tokens)
    return toxicity

def identity_toxicity_table(identities,embeddings,model,add=None):
    words = []
    toxicities = []
    for word in sorted(identities):
        word = add + word if add else word
        word = word.lower()
        toxicities.append(text_to_toxicity(model, embeddings,word.lower()))
        words.append(word)
    return zip(words,toxicities)

def load_lexicon(filename):
    lexicon = []
    with open(filename) as infile:
        for line in infile:
            line = line.rstrip()
            if line and not line.startswith(';'):
                lexicon.append(line)
    return lexicon

def generate_train_test_set(model,targets,sentences,dim=300):
    vectors = np.zeros((len(sentences),dim))
    labels = np.zeros((len(sentences)))
    count=0
    for i,sentence in enumerate(sentences):
        words = filter(lambda x: x in model,[x.lower() for x in sentence.split()])
        if len(words)>0:
            vectors[count,:] = np.mean(model[words],axis = 0)
            labels[count] = targets[i] 
            count+=1
    print count, " sentences in embeddings, ", len(sentences) - count, " sentences not in embeddings"
    return train_test_split(vectors, labels, test_size=0.1, random_state=0)

def remove_component(matrix, row_index):
    return np.delete(matrix,row_index,1)

# Deprecated. Or at least not used right now.
def remove_principle_component(X_matrix, component_index):
    u, s, vt = np.linalg.svd(X_matrix, full_matrices=False)
    u_new = np.delete(u, component_index, 1)    
    s_new = np.delete(s, component_index) 
    vt_new = np.delete(vt, component_index, 0)
    return np.mat(u_new)* np.diag(s_new) * np.mat(vt_new)

# Deprecated. Or at least not used right now.
def leave_one_out(X_matrix):
    print("leaving one out...")
    list_of_Xs = []
    for i in range(X_matrix.shape[1]):
        X_without_i = remove_principle_component(X_matrix, i)
        list_of_Xs.append(np.array(X_without_i))
    return list_of_Xs

def leave_one_out_efficiently(X_matrix, num_top_components_to_remove=float('inf')):
    list_of_Xs = []
    u, s, vt = np.linalg.svd(X_matrix, full_matrices=False)
    
    for i in range(min(X_matrix.shape[1], num_top_components_to_remove)):
        u_new = np.delete(u, i, 1)    
        s_new = np.delete(s, i) 
        vt_new = np.delete(vt, i, 0)
        X_without_i = np.mat(u_new)* np.diag(s_new) * np.mat(vt_new)
        list_of_Xs.append(np.array(X_without_i))
    return list_of_Xs

##############################################################################################################################
''''
Fair Regularization Text Classification
'''
###############################################################################################################################
class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=5000,val=None,X_tils =[],lamb=.1,reg_coeff=[]):
        self.lr = lr
        self.num_iter = num_iter
        self.val = val
        self.X_tils = X_tils
        self.lamb = lamb
        self.reg_coeff = reg_coeff
   
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def calculate_fairness_reg_grad(self,theta):
        fair_reg_sum = 0
        for i,X_til in enumerate(self.X_tils):
            l = np.shape(X_til)[0]
            z_til = np.dot(X_til, theta)
            h_til = self.__sigmoid(z_til)
            T = np.sum(h_til)
            fair_reg_sum+=self.reg_coeff[i]*(np.sum(X_til.T*h_til*(1-h_til)*(np.log(l*h_til)+1),axis=1))
        return fair_reg_sum
    
    def fit(self, X, y):
        # weights initialization
        self.theta = np.zeros(X.shape[1])
        RNSB = []
        loss = []
        validation_scores = []
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            r = self.calculate_fairness_reg_grad(self.theta)
            grad_loss = np.dot(X.T, (h - y)) / y.size
            #logistic loss + L2 regularization + sum(fair_reg)
            if(i % 1000 == 0):
                print np.sum(np.abs(grad_loss)), np.sum(np.abs(2*self.lamb*self.theta)), np.sum(np.abs(r))
            gradient = grad_loss + 2*self.lamb*self.theta+ r
            self.theta -= self.lr * gradient
            RNSB.append(self.validation_fairness())
            loss.append(self.__loss(h, y))
            validation_scores.append(self.validation(self.val[0],self.val[1]))
            if(i % 1000 == 0):
                z = np.dot(X, self.theta)
                h = self.__sigmoid(z)
                print 'loss: ' ,self.__loss(h, y)
                if self.val:
                    print 'validation MAp: ', self.validation(self.val[0],self.val[1])
                if len(self.X_tils)>0:
                    print 'RNSB: ', self.validation_fairness()
        return (RNSB,loss,validation_scores)    
    
    def predict_prob(self, X):
        return self.__sigmoid(np.dot(X, self.theta))
    
    def predict_proba(self, X):
        return [1-self.__sigmoid(np.dot(X, self.theta)),self.__sigmoid(np.dot(X, self.theta))]
    
    def predict(self, X, threshold):
        return self.predict_prob(X) >= threshold
    
    def validation(self,X,Y):
        return np.mean(self.predict(X,.5)==Y)
    
    def validation_fairness(self):
        vals = []
        for Xtil in self.X_tils:
            probs = self.predict_prob(Xtil)
            probs = probs/np.sum(probs)
            uniform = np.ones(len(probs))*1./len(probs)
            vals.append((probs * np.log(probs/uniform)).sum())
        return vals
    def print_val_fairness(self):
        vals = []
        for Xtil in self.X_tils:
            probs = self.predict_prob(Xtil)
            probs = probs/np.sum(probs)
            print probs
            uniform = np.ones(len(probs))*1./len(probs)
            vals.append((probs * np.log(probs/uniform)).sum())
        print vals
        
##############################################################################################################################
'''
adversarial debiasing: some code is derived from https://colab.research.google.com/notebooks/ml_fairness/adversarial_debiasing.ipynb
'''
#################################################################################################################################
def load_word2vec_format(f, max_num_words=None):
  """Loads word2vec data from a file handle.

  Similar to gensim.models.keyedvectors.KeyedVectors.load_word2vec_format
  but takes a file handle as input rather than a filename. This lets us use
  GFile. Also only accepts binary files.

  Args:
    f: file handle
    max_num_words: number of words to load. If None, load all.

  Returns:
    Word2vec data as keyedvectors.EuclideanKeyedVectors.
  """
  header = f.readline()
  vocab_size, vector_size = (
      int(x) for x in header.rstrip().split())  # throws for invalid file format
  print "vector_size =  %d" % vector_size
  result = gensim.models.keyedvectors.EuclideanKeyedVectors()
  num_words = 0
  result.vector_size = vector_size
  result.syn0 = np.zeros((vocab_size, vector_size), dtype=np.float32)
  
  def add_word(word, weights):
    word_id = len(result.vocab)
    if word in result.vocab:
      print("duplicate word '%s', ignoring all but first", word)
      return
    result.vocab[word] = gensim.models.keyedvectors.Vocab(
        index=word_id, count=vocab_size - word_id)
    result.syn0[word_id] = weights
    result.index2word.append(word)

  if max_num_words and max_num_words < vocab_size:
    num_embeddings = max_num_words
  else:
    num_embeddings = vocab_size
  print "Loading %d embeddings" % num_embeddings
  
  binary_len = np.dtype(np.float32).itemsize * vector_size
  for _ in xrange(vocab_size):
    # mixed text and binary: read text first, then binary
    word = []
    while True:
      ch = f.read(1)
      if ch == b' ':
        break
      if ch == b'':
        raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
      if ch != b'\n':  # ignore newlines in front of words (some binary files have)
        word.append(ch)
    word = gensim.utils.to_unicode(b''.join(word), encoding='utf-8', errors='strict')
    weights = np.frombuffer(f.read(binary_len), dtype=np.float32)
    add_word(word, weights)
    num_words = num_words + 1
    if max_num_words and num_words == max_num_words:
      break
  if result.syn0.shape[0] != len(result.vocab):
    print(
        "duplicate words detected, shrinking matrix size from %i to %i",
        result.syn0.shape[0], len(result.vocab))
  result.syn0 = np.ascontiguousarray(result.syn0[:len(result.vocab)])
  assert (len(result.vocab), vector_size) == result.syn0.shape

  print("loaded %s matrix", result.syn0.shape)
  return result

def print_knn(client, v, k):
  print "%d closest neighbors to A-B+C:" % k
  for neighbor, score in client.similar_by_vector(
      v.flatten().astype(float), topn=k):
    print "%s : score=%f" % (neighbor, score)
def load_analogies(filename):
  """Loads analogies.

  Args:
    filename: the file containing the analogies.

  Returns:
    A list containing the analogies.
  """
  analogies = []
  with open(filename, "r") as fast_file:
    for line in fast_file:
      line = line.strip()
      # in the analogy file, comments start with :
      if line[0] == ":":
        continue
      words = line.split()
      # there are no misformatted lines in the analogy file, so this should
      # only happen once we're done reading all analogies.
      if len(words) != 4:
        print "Invalid line: %s" % line
        continue
      analogies.append(words)
  print "loaded %d analogies" % len(analogies)
  return analogies

def _np_normalize(v):
  """Returns the input vector, normalized."""
  return v / np.linalg.norm(v)


def load_vectors(client, analogies):
  """Loads and returns analogies and embeddings.

  Args:
    client: the client to query.
    analogies: a list of analogies.

  Returns:
    A tuple with:
    - the embedding matrix itself
    - a dictionary mapping from strings to their corresponding indices
      in the embedding matrix
    - the list of words, in the order they are found in the embedding matrix
  """
  words_unfiltered = set()
  for analogy in analogies:
    words_unfiltered.update(analogy)
  print "found %d unique words" % len(words_unfiltered)

  vecs = []
  words = []
  index_map = {}
  for word in words_unfiltered:
    try:
      vecs.append(_np_normalize(client.word_vec(word)))
      index_map[word] = len(words)
      words.append(word)
    except KeyError:
      print "word not found: %s" % word
  print "words not filtered out: %d" % len(words)

  return np.array(vecs), index_map, words

def load_vectors_test(embeddings,analogies):
    X = []
    y = []
    l=[]
    for i in analogies:
        if all(j.lower() in embeddings for j in i):
            B = _np_normalize(embeddings[i[1]])
            C = _np_normalize(embeddings[i[2]])
            A = _np_normalize(embeddings[i[0]])
            D = _np_normalize(embeddings[i[3]])
            X.append(B+C-A)
            y.append(D)
            l.append(i[3])
    return (X,y,l)

def analogy_accuracy(X,y,embeddings,n = 5,samples=100,model=None):
    correct = 0
    count = 0
    for i in np.random.choice(np.shape(X)[0],samples,replace=False):
        count+=1
        if count % 10 ==0:
            pass
            print(count/float(samples))
        vec = model(X[i]) if model else X[i]
        for j in embeddings.similar_by_vector(vec,topn=n):
            try:
                if y[i] == str(j[0]): 
                    correct+=1
                    break
            except:
                pass
    return float(correct)/samples
from scipy import stats
from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
def similarity(w1, w2):
    return np.dot(matutils.unitvec(w1), matutils.unitvec(w2))
def evaluate_word_pairs(model, pairs, debias,delimiter='\t', restrict_vocab=300000, case_insensitive=True,
                            dummy4unknown=True):
        """
        Compute correlation of the model with human similarity judgments. `pairs` is a filename of a dataset where
        lines are 3-tuples, each consisting of a word pair and a similarity value, separated by `delimiter`.
        An example dataset is included in Gensim (test/test_data/wordsim353.tsv). More datasets can be found at
        http://technion.ac.il/~ira.leviant/MultilingualVSMdata.html or https://www.cl.cam.ac.uk/~fh295/simlex.html.

        The model is evaluated using Pearson correlation coefficient and Spearman rank-order correlation coefficient
        between the similarities from the dataset and the similarities produced by the model itself.
        The results are printed to log and returned as a triple (pearson, spearman, ratio of pairs with unknown words).

        Use `restrict_vocab` to ignore all word pairs containing a word not in the first `restrict_vocab`
        words (default 300,000). This may be meaningful if you've sorted the vocabulary by descending frequency.
        If `case_insensitive` is True, the first `restrict_vocab` words are taken, and then case normalization
        is performed.

        Use `case_insensitive` to convert all words in the pairs and vocab to their uppercase form before
        evaluating the model (default True). Useful when you expect case-mismatch between training tokens
        and words pairs in the dataset. If there are multiple case variants of a single word, the vector for the first
        occurrence (also the most frequent if vocabulary is sorted) is taken.

        Use `dummy4unknown=True` to produce zero-valued similarities for pairs with out-of-vocabulary words.
        Otherwise (default False), these pairs are skipped entirely.
        """
        ok_vocab = [(w, model.vocab[w]) for w in model.index2word[:restrict_vocab] if w in model]
        ok_vocab = dict((w.upper(), v) for w, v in reversed(ok_vocab)) if case_insensitive else dict(ok_vocab)

        similarity_gold = []
        similarity_model = []
        oov = 0

        original_vocab = model.vocab
        model.vocab = ok_vocab

        for line_no, line in enumerate(utils.smart_open(pairs)):
            line = utils.to_unicode(line)
            if line.startswith('#'):
                # May be a comment
                continue
            else:
                try:
                    if case_insensitive:
                        a, b, sim = [word.upper() for word in line.split(delimiter)]
                    else:
                        a, b, sim = [word for word in line.split(delimiter)]
                    sim = float(sim)
                except:
                    continue
                if a not in ok_vocab or b not in ok_vocab:
                    oov += 1
                    if dummy4unknown:
                        similarity_model.append(0.0)
                        similarity_gold.append(sim)
                        continue
                    else:
                        continue
                similarity_gold.append(sim)  # Similarity from the dataset
                similarity_model.append(similarity(debias(model[a]), debias(model[b])))  # Similarity from the model
        model.vocab = original_vocab
        spearman = stats.spearmanr(similarity_gold, similarity_model)
        pearson = stats.pearsonr(similarity_gold, similarity_model)
        oov_ratio = float(oov) / (len(similarity_gold) + oov) * 100
        return pearson, spearman, oov_ratio
    
    
def tf_normalize(x):
  """Returns the input vector, normalized.

  A small number is added to the norm so that this function does not break when
  dealing with the zero vector (e.g. if the weights are zero-initialized).

  Args:
    x: the tensor to normalize
  """
  return x / (tf.norm(x) + np.finfo(np.float32).tiny)


class AdversarialEmbeddingModel(object):
  """A model for doing adversarial training of embedding models."""

  def __init__(self, client,
               data_p, embed_dim, projection,
               projection_dims, pred,verbose=False):
    """Creates a new AdversarialEmbeddingModel.

    Args:
      client: The (possibly biased) embeddings.
      data_p: Placeholder for the data.
      embed_dim: Number of dimensions used in the embeddings.
      projection: The space onto which we are "projecting".
      projection_dims: Number of dimensions of the projection.
      pred: Prediction layer.
    """
    # load the analogy vectors as well as the embeddings
    self.client = client
    self.data_p = data_p
    self.embed_dim = embed_dim
    self.projection = projection
    self.projection_dims = projection_dims
    self.pred = pred
    self.verbose = verbose

  def nearest_neighbors(self, sess, in_arr,
                        k):
    """Finds the nearest neighbors to a vector.

    Args:
      sess: Session to use.
      in_arr: Vector to find nearest neighbors to.
      k: Number of nearest neighbors to return
    Returns:
      List of up to k pairs of (word, score).
    """
    v = sess.run(self.pred, feed_dict={self.data_p: in_arr})
    return self.client.similar_by_vector(v.flatten().astype(float), topn=k)

  def write_to_file(self, sess, f):
    """Writes a model to disk."""
    np.savetxt(f, sess.run(self.projection))

  def read_from_file(self, sess, f):
    """Reads a model from disk."""
    loaded_projection = np.loadtxt(f).reshape(
        [self.embed_dim, self.projection_dims])
    sess.run(self.projection.assign(loaded_projection))

    
  def fit(self,
          sess,
          data,
          data_p,
          labels,
          labels_p,
          protect,
          protect_p,
          gender_direction,
          pred_learning_rate,
          protect_learning_rate,
          protect_loss_weight,
          num_steps,
          batch_size,
          debug_interval=1000):
    """Trains a model.

    Args:
      sess: Session.
      data: Features for the training data.
      data_p: Placeholder for the features for the training data.
      labels: Labels for the training data.
      labels_p: Placeholder for the labels for the training data.
      protect: Protected variables.
      protect_p: Placeholder for the protected variables.
      gender_direction: The vector from find_gender_direction().
      pred_learning_rate: Learning rate for predicting labels.
      protect_learning_rate: Learning rate for protecting variables.
      protect_loss_weight: The constant 'alpha' found in
          debias_word_embeddings.ipynb.
      num_steps: Number of training steps.
      batch_size: Number of training examples in each step.
      debug_interval: Frequency at which to log performance metrics during
          training.
    """
  
    ##########################
    feed_dict = {
        data_p: data,
        labels_p: labels,
        protect_p: protect    }
    
    # define the prediction loss
    pred_loss = tf.losses.mean_squared_error(labels_p, self.pred)
#     pred_loss = tf.losses.absolute_difference(labels_p, self.pred)

    # compute the prediction of the protected variable.
    # The "trainable"/"not trainable" designations are for the predictor. The
    # adversary explicitly specifies its own list of weights to train.
    protect_weights = tf.get_variable(
        "protect_weights", [self.embed_dim, 1], trainable=False)
    
    protect_pred = tf.matmul(self.pred, protect_weights)#changr
    protect_loss = tf.losses.mean_squared_error(protect_p, protect_pred)

    pred_opt = tf.train.AdamOptimizer(pred_learning_rate)
    protect_opt = tf.train.AdamOptimizer(protect_learning_rate)

    protect_grad = {v: g for (g, v) in pred_opt.compute_gradients(protect_loss)}
    pred_grad = []

    # applies the gradient expression found in the document linked
    # at the top of this file.
    for (g, v) in pred_opt.compute_gradients(pred_loss):
      unit_protect = tf_normalize(protect_grad[v])
      # the two lines below can be commented out to train without debiasing
      g -= tf.reduce_sum(g * unit_protect) * unit_protect
      g -= protect_loss_weight * protect_grad[v]
      pred_grad.append((g, v))
      pred_min = pred_opt.apply_gradients(pred_grad)

    # compute the loss of the protected variable prediction.
    protect_min = protect_opt.minimize(protect_loss, var_list=[protect_weights])
  # Parameters

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    ###############################################################################
    step = 0
    while step < num_steps:
      # pick samples at random without replacement as a minibatch
      ids = np.random.choice(len(data), batch_size, False)
      data_s, labels_s, protect_s = data[ids], labels[ids], protect[ids]
      sgd_feed_dict = {
          data_p: data_s,
          labels_p: labels_s,
          protect_p: protect_s,
      }

      if not step % debug_interval and self.verbose:
        metrics = [pred_loss, protect_loss, self.projection]
        metrics_o = sess.run(metrics, feed_dict=feed_dict)
        pred_loss_o, protect_loss_o, proj_o = metrics_o
#         log stats every so often: number of steps that have passed,
#         prediction loss, adversary loss
        print("step: %d; pred_loss_o: %f; protect_loss_o: %f" % (step,
                     pred_loss_o, protect_loss_o))
        for i in range(proj_o.shape[1]):
          print("proj_o: %f; dot(proj_o, gender_direction): %f)" %
                       (np.linalg.norm(proj_o[:, i]),
                       np.dot(proj_o[:, i].flatten(), gender_direction)))
      sess.run([pred_min, protect_min], feed_dict=sgd_feed_dict)
      step += 1
      
def filter_analogies(analogies,
                     index_map):
  filtered_analogies = []
  for analogy in analogies:
    if filter(index_map.has_key, analogy) != analogy:
      print "at least one word missing for analogy: %s" % analogy
    else:
      filtered_analogies.append(map(index_map.get, analogy))
  return filtered_analogies

def make_data(
    analogies, embed,
    gender_direction):
  """Preps the training data.

  Args:
    analogies: a list of analogies
    embed: the embedding matrix from load_vectors
    gender_direction: the gender direction from find_gender_direction

  Returns:
    Three numpy arrays corresponding respectively to the input, output, and
    protected variables.
  """
  data = []
  labels = []
  protect = []
  for analogy in analogies:
    # the input is just the word embeddings of the first three words
    data.append(embed[analogy[:3]])
    # the output is just the word embeddings of the last word
    labels.append(embed[analogy[3]])
    # the protected variable is the gender component of the output embedding.
    # the extra pair of [] is so that the array has the right shape after
    # it is converted to a numpy array.
    protect.append([np.dot(embed[analogy[3]], gender_direction)])
  # Convert all three to numpy arrays, and return them.
  return tuple(map(np.array, (data, labels, protect)))