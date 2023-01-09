import datetime
import os
import sys

from absl import app, flags
from loguru import logger
import discord
import tensorflow as tf

# Disable GPU processing for the model's inference
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

FLAGS = flags.FLAGS

flags.DEFINE_string('model_path', '', 'Filesystem path to a tensorflow model to run over each message')
flags.DEFINE_string('discord_token', '', 'The Discord bot token generated for this particular bot')
flags.DEFINE_string('log_level', 'INFO', 'Loguru log level to output to stderr')

def main(argv):
    ensure_configuration_or_die()
    configure_logging()
    logger.info("Initializing bot")
    model = load_model()
    client = setup_bot(model)
    client.run(FLAGS.discord_token)

def configure_logging():
    # Remove the default stderr handler
    logger.remove()
    logger.add(sys.stderr, level=FLAGS.log_level)

def ensure_configuration_or_die():
    if FLAGS.discord_token == "":
        logger.critical("discord_token is required configuration")
        exit()
    if FLAGS.model_path == "":
        logger.critical("model_path is required configuration")
        exit()

def load_model():
    logger.info(f"Loading model from path {FLAGS.model_path}")
    return tf.keras.models.load_model(FLAGS.model_path)

def setup_bot(model):
    intents = discord.Intents.default()
    intents.message_content = True

    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        logger.info(f'We have logged in as {client.user}')
        guild_names = list(map(lambda x: x.name, client.guilds))
        logger.info(f'We are associated with {guild_names}')

    @client.event
    async def on_message(message):
        logger.debug(f'Received message in #{message.channel}: {message.content}')
        if str(message.channel) not in ["general", "development", "bot-test", "hardware-dev", "configurator"] and not hasattr(message.channel, 'parent_id'):
            logger.debug(f'Ignoring message in unmonitored channel {message.channel}')
            return

        # Ignore our own messages
        if message.author == client.user:
            return

        # Calculate how old the user is, and a threshold for the model
        user_days_old = (datetime.datetime.now(datetime.timezone.utc) - message.author.joined_at).days
        if user_days_old < 3:
            threshold = 0.80
        elif user_days_old < 14:
            threshold = 0.90
        elif user_days_old < 30:
            threshold = 0.925
        elif user_days_old < 180:
            threshold = 0.95
        elif user_days_old < 360:
            threshold = 0.99
        else:
            threshold = 1.0
        friendly_threshold = round(threshold * 100)
        logger.debug(f'User {message.author} has been here for {user_days_old} days, setting threshold={threshold}')

        prediction_results = model.predict([message.content], verbose=0)
        confidence = prediction_results[0][0]
        friendly_confidence = round(confidence * 100)

        # Send a reply if it triggers the model
        if confidence > threshold:
                logger.debug(f'Message is likely a help question with {friendly_confidence}%{friendly_threshold}% confidence')
                await message.channel.send(f'Hey there! Our bot detected that this may be a help question, which you\'ll get a better response to in #help-and-support, please feel free to ask your question there!', reference=message)
        else:
                logger.debug(f'Message is likely not a help question with {friendly_confidence}%/{friendly_threshold}% confidence')
        
        # Allow for test messages
        if client.user in message.mentions:
                await message.channel.send(f'{friendly_confidence}%/{friendly_threshold}% confidence', reference=message)
                logger.debug(f'Test message received with {friendly_confidence}%/{friendly_threshold}% confidence')

    return client


if __name__ == "__main__":
    app.run(main)
