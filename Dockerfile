# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /code

# Copy the requirements file into the container at /code
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /code
COPY . .

# Make port 80 available to the world outside this container (standard for HF Docker spaces, though the bot doesn't use HTTP)
# EXPOSE 80
# Health check (optional but good practice for HF Spaces)
# HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 CMD python -c "import socket; socket.create_connection(('localhost', 80))" || exit 1
EXPOSE 7860


# Define environment variable (optional, can use secrets instead)
# ENV NAME World

# Run bot.py when the container launches
CMD ["python", "bot.py"]