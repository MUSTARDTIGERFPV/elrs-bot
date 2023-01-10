import datetime
import os
import sys
import shelve

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
flags.DEFINE_string('db_path', 'bot.shelve', 'Filesystem path to the database we want to use')
flags.DEFINE_integer('rewarn_threshold_hours', 3, 'How many hours to wait before rewarning the same user')

def main(argv):
    ensure_configuration_or_die()
    configure_logging()
    logger.info("Initializing bot")
    model = load_model()
    db = setup_database()
    client = setup_bot(model, db)
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

def setup_database():
    return shelve.open(FLAGS.db_path)

def load_model():
    logger.info(f"Loading model from path {FLAGS.model_path}")
    return tf.keras.models.load_model(FLAGS.model_path)

def setup_bot(model, db):
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

        should_send_response = True

        # Ignore our own messages
        if message.author == client.user:
            should_send_response = False

        # Calculate how old the user is, and a threshold for the model
        user_days_old = days_since(message.author.joined_at)
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

        if confidence < threshold:
            should_send_response = False
            logger.info(f'Dropping message because we didn\'t meet threshold. {friendly_confidence} < {friendly_threshold}')

        # Handle if we've told this user recently
        friendly_author = str(message.author)
        if friendly_author in db:
            last_warning = hours_since(db[friendly_author])
            if last_warning < FLAGS.rewarn_threshold_hours:
                should_send_response = False
                logger.info(f'Dropping message because user was warned recently at {last_warning}')
                
        supported_channel_names = ["general", "development", "bot-test", "hardware-dev", "configurator"]
        # The list of parent channels we want to respond to threads in
        thread_channels = [596350022191415320] # Currently just #general
        if str(message.channel) not in supported_channel_names or (hasattr(message.channel, 'parent_id') and message.channel.parent_id not in thread_channels):
            should_send_response = False
            logger.info(f'Dropping message in unmonitored channel {message.channel}')

        if should_send_response:
                logger.info(f'Sending response to user {messsage.author} in {message.channel}')
                # Send a message to the user
                await message.channel.send(f'Hey there! Our bot is {friendly_confidence}% sure that this is a help question, which you\'ll get a better response to in <#798006228450017290>. Please feel free to ask your question there!', reference=message)
                # Set the time we last sent this user a message
                db[str(message.author)] = datetime.now(datetime.timezone.utc)
        else:
                logger.info(f'Not sending response to user {message.author} in {message.channel}')
        
        # Allow for test messages
        if client.user in message.mentions:
                await message.channel.send(f'{friendly_confidence}%/{friendly_threshold}% confidence', reference=message)
                logger.debug(f'Test message received with {friendly_confidence}%/{friendly_threshold}% confidence')

    return client

def hours_since(d):
    return (datetime.datetime.now(datetime.timezone.utc) - d).total_seconds() / (60 * 60)
def days_since(d):
    return hours_since(d) / 24


if __name__ == "__main__":
    app.run(main)
