import numpy as np
import tensorflow as tf
# from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.applications.vgg19 import preprocess_input, VGG19
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam

tf.compat.v1.enable_eager_execution()
model = VGG19(include_top = False, weights='imagenet')
model.trainable = False
print(model.summary())

content_image = cv2.imread("C:\\Users\\ASUS\\Desktop\\python_prac\\louvre.jpg")
content_image = cv2.cvtColor(content_image, cv2.COLOR_BGR2RGB)
content_image = cv2.resize(content_image, (300, 300))
content_image = content_image.reshape(-1, 300, 300, 3)
content_image = preprocess_input(content_image)
# plt.imshow(np.squeeze(content_image))
# plt.show()

style_image = cv2.imread("C:\\Users\\ASUS\\Desktop\\python_prac\\monet_800600.jpg")
style_image = cv2.cvtColor(style_image, cv2.COLOR_BGR2RGB)
style_image = cv2.resize(style_image, (300, 300))
style_image = style_image.reshape(-1, 300, 300, 3)
style_image = preprocess_input(style_image)

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

content_layers = 'block5_conv1'

#Defining the input and output for the model
content_model = Model(inputs = model.input, outputs = model.get_layer(content_layers).output)
style_models = [Model(inputs = model.input, outputs = model.get_layer(i).output) for i in style_layers]


def content_cost(content, generated):
    a_C = content_model.inputs = content
    a_G = content_model.inputs = generated
    J_content = tf.reduce_mean(tf.square(a_C - a_G))
    return J_content

def gram_martrix(A):
    m, n_H, n_W, n_C  = A.shape
    A = tf.reshape(A, [n_H * n_W, n_C])
    return tf.matmul(A, tf.transpose(A))

def style_cost(style, generated):
    current_cost = []
    for style_model in style_models:
        a_S = style_model.inputs = style
        a_G = style_model.inputs = generated
        G_S = gram_martrix(a_S)
        G_G = gram_martrix(a_G)
        current_cost.append(tf.reduce_mean(tf.square(G_S, G_G)))

    J_style = 0.2 * current_cost[0] + 0.2 * current_cost[1] + 0.2 * current_cost[2] + 0.2 * current_cost[3] + 0.2 * current_cost[4]
    return J_style

def train(content_image, style_image, iterations, alpha, beta, lr):

    content = content_image
    style = style_image
    generated = tf.Variable(content)
    opt = Adam(learning_rate=lr)
    J_min = 1e12 + 0.1
    for i in range(iterations + 1):
        with tf.GradientTape() as g:
            J_content = content_cost(content, generated)
            J_style = style_cost(style, generated)
            J_cost = alpha * J_content + beta * J_style

        grads = g.gradient(J_cost, generated)
        opt.apply_gradients(grads, generated)

        if J_cost < J_min:
            J_min = J_cost  # picks the generated image with the lowest J_cost
            best_pic = generated.numpy()

        if i % 20 == 0:
            print('cost at iteration ' + str(i) + ' = ' + str(J_cost.numpy()))


    return best_pic


itr = 200
alpha = 10
beta = 40
lr = 2.0


train(content_image, style_image, itr, alpha, beta, lr)