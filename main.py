import collections
import csv
import datetime
import os
import re
import shelve
import sys

from absl import app, flags
from loguru import logger
import discord
import tensorflow as tf
import manhole

# Disable GPU processing for the model's inference
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

FLAGS = flags.FLAGS

flags.DEFINE_string('model_path', '', 'Filesystem path to a tensorflow model to run over each message')
flags.DEFINE_string('discord_token', '', 'The Discord bot token generated for this particular bot')
flags.DEFINE_string('log_level', 'INFO', 'Loguru log level to output to stderr')
flags.DEFINE_string('db_path', 'bot.shelve', 'Filesystem path to the database we want to use')
flags.DEFINE_integer('rewarn_threshold_hours', 3, 'How many hours to wait before rewarning the same user')
flags.DEFINE_integer('log_buffer_size', 50, 'How many log lines to keep in memory')
flags.DEFINE_string('only_run_in_guild', '', 'If set, the bot will only run in the named Discord server')
flags.DEFINE_string('vote_filename', 'votes.csv', 'Filesystem path to a text file where each line is a new entry for a user voting up or down a response')
#flags.DEFINE_integer('log_channel_id', 1062766435639758939, 'Discord channel ID to send logs in')
flags.DEFINE_integer('log_channel_id', 1067470822605864991, 'Discord channel ID to send logs in')

def main(argv):
    ensure_configuration_or_die()
    log_buffer = custom_logger()
    configure_logging(log_buffer)
    logger.info("Initializing bot")
    model = load_model()
    db = setup_database()
    client = setup_bot(model, db, log_buffer)
    manhole.install(patch_fork=False, daemon_connection=True, locals={"client": client, "model": model})
    client.run(FLAGS.discord_token)

class BufferLogger:
    def __init__(self):
        self.d = collections.deque(maxlen=FLAGS.log_buffer_size)
    def write(self, msg):
        self.d.append(msg.strip())
    def dump(self):
        return "```" + "\n".join(self.d) + "```"
    def dump_up_to_characters(self, n):
        total_length = float('inf')
        start_index = 0
        while total_length > n:
            total_length = sum(map(lambda x: len(x) + 1, list(self.d)[start_index:]))
            start_index = start_index + 1
        return "\n".join(list(self.d)[start_index:])
        

def custom_logger():
    return BufferLogger()

def configure_logging(log_buffer):
    # Remove the default stderr handler
    logger.remove()
    logger.add(sys.stderr, level=FLAGS.log_level)
    logger.add(log_buffer, level=FLAGS.log_level)
    logger.add('logs/helper_{time}.log', rotation="1 day", retention="1 week", level=FLAGS.log_level)

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
    vote_file = open(FLAGS.vote_filename, 'a+')
    votewriter = csv.writer(vote_file, delimiter=',')

    @client.event
    async def on_ready():
        logger.info(f'We have logged in as {client.user}')
        guild_names = list(map(lambda x: x.id, client.guilds))
        logger.info(f'We are associated with guilds {guild_names}')

    @client.event
    async def on_raw_reaction_add(reaction):
        channel = client.get_channel(reaction.channel_id)
        response_message = await channel.fetch_message(reaction.message_id)
        logger.debug(f'Received reaction "{reaction.emoji.name}" add event on #{channel} message ID {response_message.id}')
        original_message_content = ""
        # Process reactions when they're to our own messages
        if response_message.author == client.user:
            # This was a response to a user message
            if response_message.reference is not None:
                # Find the message this was in response to
                original_channel = client.get_channel(response_message.reference.channel_id)
                original_message = await original_channel.fetch_message(response_message.reference.message_id)
                logger.debug(f'Received reaction add event: {reaction.emoji.name} on {original_message.content}')
                original_message_content = re.sub(r'\n', r' ', original_message.content)
                # Clean message content
                if client.user.mentioned_in(original_message) and "test" in original_message_content:
                    # Remove @bot test from string
                    original_message_content = " ".join(original_message_content.split(" ")[2:])
            # This was a log message
            elif response_message.channel.id == FLAGS.log_channel_id:
                regex = re.compile(r'\"(.*)\"')
                # Strip the filler text
                original_message_content = regex.findall(response_message.content)[0]
            # Remove backslashes
            original_message_content = re.sub(r'\\', '', original_message_content)

            if reaction.emoji.name == '👍':
                logger.info(f'Storing positive reaction for {original_message_content}')
                votewriter.writerow([1, original_message_content])
            if reaction.emoji.name == '👎':
                logger.info(f'Storing negative reaction for {original_message_content}')
                votewriter.writerow([0, original_message_content])
            vote_file.flush()

    @client.event
    async def on_message(message):
        message_content_stripped = re.sub(r'\n', r' ', message.content)
        logger.debug(f'Received message in {message.guild} #{message.channel}: {message_content_stripped}')

        should_send_response = True

        # Watch for dry-run mode
        if FLAGS.only_run_in_guild != "" and str(message.guild) != FLAGS.only_run_in_guild:
            logger.info(f'Dropping message because we\'re in a dry run mode')
            should_send_response = False

        # Ignore our own messages
        if message.author == client.user:
            logger.debug('Dropping message from self')
            should_send_response = False
        if message.author.bot:
            logger.debug('Dropping message from bot')
            return

        # Calculate how old the user is, and a threshold for the model
        user_days_old = round(days_since(message.author.joined_at))
        if user_days_old < 3:
            threshold = 0.80
        elif user_days_old < 14:
            threshold = 0.825
        elif user_days_old < 30:
            threshold = 0.850
        elif user_days_old < int(365*0.5):
            threshold = 0.875
        elif user_days_old < int(365*1.5):
            threshold = 0.925
        else:
            threshold = 1.0

        friendly_threshold = friendly_percentage(threshold)
        logger.debug(f'User {message.author} has been here for {user_days_old} days, setting threshold={threshold}')

        prediction_results = model.predict([message.content], verbose=0)
        confidence = prediction_results[0][0]
        friendly_confidence = friendly_percentage(confidence)
        logger.debug(f'Message confidence {friendly_confidence}% / {friendly_threshold}%')

        if confidence < threshold:
            should_send_response = False
            logger.debug(f'Dropping message because we didn\'t meet threshold. {friendly_confidence} < {friendly_threshold}')
        else:
            logger.debug(f'Model threshold reached: {friendly_confidence} / {friendly_threshold} for {message.content}')
            

        # Handle if we've told this user recently
        friendly_author = str(message.author)
        if friendly_author in db:
            last_warning = hours_since(db[friendly_author])
            if last_warning < FLAGS.rewarn_threshold_hours:
                should_send_response = False
                logger.debug(f'Dropping message because user was warned recently at {last_warning}')
                
        # Reject messages for channels we don't want to monitor
        supported_channel_names = ["general", "development", "bot-test", "hardware-dev", "configurator"]
        thread_channels = [596350022191415320] # The list of parent channels for which we want to respond to threads; currently just #general
        if str(message.channel) not in supported_channel_names or (hasattr(message.channel, 'parent_id') and message.channel.parent_id not in thread_channels):
            should_send_response = False
            logger.debug(f'Dropping message in unmonitored channel {message.channel}')

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
                # Discord limits messages to 2000 characters, we use 6 for the backticks
                response_length = 2000 - 6
                response_content = discord.utils.escape_mentions(log_buffer.dump_up_to_characters(response_length))
                await message.channel.send('```' + response_content + '```', reference=message)
            elif command == 'ping' or command == 'status':
                await message.channel.send(f'''
                Running as PID {os.getpid()} using model {FLAGS.model_path}
                ''')
            elif command == 'help':
                await message.channel.send('''
                Commands:
                * test - tests the remainder of the message against the model
                * log - shows the last few lines of the bot's log
                * help
                ''', reference=message)

        if should_send_response:
            logger.info(f'Sending response to user {message.author} in {message.guild} #{message.channel}')
            # Send a message to the user
            await message.channel.send(f'Hey there! Our bot is {friendly_confidence}% sure that this is a help question, which you\'ll get a better response to in <#798006228450017290>. Please feel free to ask your question there!\n\nDid I get this right? If so, leave me a 👍. If not, please leave me a 👎. I\'ll use your response to get better.', reference=message)
            # Set the time we last sent this user a message
            db[str(message.author)] = datetime.datetime.now(datetime.timezone.utc)
        else:
            logger.debug(f'Not sending response to user {message.author} in {message.guild} #{message.channel}')
        
        # Send final disposition message to logs thread in ExpressLRS Discord
        if message.guild.id == 596350022191415318:
            channel = client.get_channel(FLAGS.log_channel_id)
            content = discord.utils.escape_mentions(message_content_stripped)
            await channel.send(f'{message.channel.mention} \"{content}\": {friendly_confidence}% / {friendly_threshold}%')
        
    return client

def hours_since(d):
    return (datetime.datetime.now(datetime.timezone.utc) - d).total_seconds() / (60 * 60)
def days_since(d):
    return hours_since(d) / 24
def friendly_percentage(n):
    return round(n * 100, 1)


if __name__ == "__main__":
    app.run(main)
