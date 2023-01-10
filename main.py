import collections
import datetime
import os
import shelve
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
flags.DEFINE_string('db_path', 'bot.shelve', 'Filesystem path to the database we want to use')
flags.DEFINE_integer('rewarn_threshold_hours', 3, 'How many hours to wait before rewarning the same user')
flags.DEFINE_integer('log_buffer_size', 10, 'How many log lines to keep in memory')
flags.DEFINE_string('only_run_in_guild', '', 'If set, the bot will only run in the named Discord server')

def main(argv):
    ensure_configuration_or_die()
    log_buffer = custom_logger()
    configure_logging(log_buffer)
    logger.info("Initializing bot")
    model = load_model()
    db = setup_database()
    client = setup_bot(model, db, log_buffer)
    client.run(FLAGS.discord_token)

class BufferLogger:
    def __init__(self):
        self.d = collections.deque(maxlen=FLAGS.log_buffer_size)
    def write(self, msg):
        self.d.append(msg.strip())
    def dump(self):
        return "```" + "\n".join(self.d) + "```"

def custom_logger():
    return BufferLogger()

def configure_logging(log_buffer):
    # Remove the default stderr handler
    logger.remove()
    logger.add(sys.stderr, level=FLAGS.log_level)
    logger.add(log_buffer, level=FLAGS.log_level)

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

def setup_bot(model, db, log_buffer):
    intents = discord.Intents.default()
    intents.message_content = True

    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        logger.info(f'We have logged in as {client.user}')
        guild_names = list(map(lambda x: x.id, client.guilds))
        logger.info(f'We are associated with guilds {guild_names}')

    @client.event
    async def on_message(message):
        logger.debug(f'Received message in {message.guild} #{message.channel}: {message.content}')

        should_send_response = True

        # Watch for dry-run mode
        if FLAGS.only_run_in_guild != "" and str(message.guild) != FLAGS.only_run_in_guild:
            logger.info(f'Dropping message because we\'re in a dry run mode')
            should_send_response = False

        # Ignore our own messages
        if message.author == client.user:
            logger.info('Dropping message from self')
            should_send_response = False
        if message.author.bot:
            logger.info('Dropping message from bot')
            return

        # Calculate how old the user is, and a threshold for the model
        user_days_old = round(days_since(message.author.joined_at))
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
        friendly_threshold = friendly_percentage(threshold)
        logger.debug(f'User {message.author} has been here for {user_days_old} days, setting threshold={threshold}')

        prediction_results = model.predict([message.content], verbose=0)
        confidence = prediction_results[0][0]
        friendly_confidence = friendly_percentage(confidence)

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
                
        # Reject messages for channels we don't want to monitor
        supported_channel_names = ["general", "development", "bot-test", "hardware-dev", "configurator"]
        thread_channels = [596350022191415320] # The list of parent channels for which we want to respond to threads; currently just #general
        if str(message.channel) not in supported_channel_names or (hasattr(message.channel, 'parent_id') and message.channel.parent_id not in thread_channels):
            should_send_response = False
            logger.info(f'Dropping message in unmonitored channel {message.channel}')

        # Allow for test messages
        if client.user in message.mentions:
            should_send_response = False
            words = message.content.split()[1:]
            logger.debug(f'Test message received with commands: {words}')
            if len(words) == 0:
                # Do nothing
                logger.debug(f'Received empty mention message')
                return

            command = words[0].lower()
            if command == 'test':
                if len(words) > 1:
                    test_string = ' '.join(words[1:])
                    confidence = model.predict([test_string], verbose=0)[0][0]
                    friendly_confidence = friendly_percentage(confidence)
                    await message.channel.send(f'{friendly_confidence}% confidence this is a help request.\n{friendly_threshold}% required to send a message because your user is {user_days_old} days old', reference=message)
                else:
                    await message.channel.send(f'Nothing to test, please send a message', reference=message)
                    
            elif command == 'log' or command == 'logs':
                await message.channel.send(log_buffer.dump(), reference=message)
            elif command == 'help':
                await message.channel.send('''
                Commands:
                * test - tests the remainder of the message against the model
                * log - shows the last few lines of the bot's log
                * help
                ''', reference=message)

        if should_send_response:
            logger.info(f'Sending response to user {messsage.author} in {message.guild} #{message.channel}')
            # Send a message to the user
            await message.channel.send(f'Hey there! Our bot is {friendly_confidence}% sure that this is a help question, which you\'ll get a better response to in <#798006228450017290>. Please feel free to ask your question there!', reference=message)
            # Set the time we last sent this user a message
            db[str(message.author)] = datetime.now(datetime.timezone.utc)
        else:
            logger.info(f'Not sending response to user {message.author} in {message.guild} #{message.channel}')
        
                

    return client

def hours_since(d):
    return (datetime.datetime.now(datetime.timezone.utc) - d).total_seconds() / (60 * 60)
def days_since(d):
    return hours_since(d) / 24
def friendly_percentage(n):
    return round(n * 100)


if __name__ == "__main__":
    app.run(main)
