from nlpmimic.models.dio_rae_parser import DioRaeParser
from nlpmimic.models.srl_gan_model import GanSemanticRoleLabeler
from nlpmimic.models.srl_naive_model import NaiveSemanticRoleLabeler
from nlpmimic.models.srl_graph_model import GraphSemanticRoleLabeler

from nlpmimic.models.vae.srl_vae_lstms import SrlLstmsAutoencoder
from nlpmimic.models.vae.srl_vae_basic import SrlBasicAutoencoder
from nlpmimic.models.vae.srl_vae_lemma import SrlLemmaAutoencoder
from nlpmimic.models.vae.srl_vae_graph import SrlGraphAutoencoder
from nlpmimic.models.vae.srl_vae_classifier import SrlVaeClassifier

from nlpmimic.models.gan.generator import SrlGanGenerator
from nlpmimic.models.gan.disc_wgan import SrlWganDiscriminator
from nlpmimic.models.gan.discriminator import SrlGanDiscriminator

