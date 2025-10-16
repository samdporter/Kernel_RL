#!/bin/bash
# Quick start script for running KRL in Docker

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print colored message
print_msg() {
    echo -e "${GREEN}[KRL Docker]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    echo "Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    print_error "Docker daemon is not running. Please start Docker first."
    exit 1
fi

# Function to show usage
usage() {
    cat << EOF
Usage: $0 [COMMAND] [OPTIONS]

Commands:
    build       Build the Docker image
    start       Start the container in the background
    stop        Stop the container
    restart     Restart the container
    shell       Start an interactive bash shell in the container
    exec        Execute a command in the container (pass command as arguments)
    logs        Show container logs
    clean       Stop and remove containers, networks, and volumes
    help        Show this help message

Examples:
    $0 build                    # Build the Docker image
    $0 start                    # Start the container
    $0 shell                    # Get an interactive shell
    $0 exec pytest              # Run tests
    $0 exec python run_deconv.py --help  # Run the main script
    $0 logs                     # View logs
    $0 stop                     # Stop the container
    $0 clean                    # Complete cleanup

For more information, see DOCKER.md
EOF
}

# Parse command
COMMAND=${1:-help}

case "$COMMAND" in
    build)
        print_msg "Building Docker image..."
        docker-compose build
        print_msg "Build complete!"
        ;;

    start)
        print_msg "Starting container..."
        docker-compose up -d
        print_msg "Container started! Use '$0 shell' to access it."
        ;;

    stop)
        print_msg "Stopping container..."
        docker-compose stop
        print_msg "Container stopped."
        ;;

    restart)
        print_msg "Restarting container..."
        docker-compose restart
        print_msg "Container restarted."
        ;;

    shell)
        # Check if container is running
        if ! docker-compose ps | grep -q "Up"; then
            print_warning "Container is not running. Starting it now..."
            docker-compose up -d
            sleep 2
        fi
        print_msg "Starting interactive shell..."
        docker-compose exec krl bash
        ;;

    exec)
        # Check if container is running
        if ! docker-compose ps | grep -q "Up"; then
            print_error "Container is not running. Start it with: $0 start"
            exit 1
        fi

        shift  # Remove 'exec' from arguments
        if [ $# -eq 0 ]; then
            print_error "No command provided. Usage: $0 exec <command>"
            exit 1
        fi

        print_msg "Executing: $@"
        docker-compose exec krl "$@"
        ;;

    logs)
        print_msg "Showing logs (Ctrl+C to exit)..."
        docker-compose logs -f krl
        ;;

    clean)
        print_warning "This will stop and remove all containers, networks, and volumes."
        read -p "Are you sure? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_msg "Cleaning up..."
            docker-compose down -v
            print_msg "Cleanup complete!"
        else
            print_msg "Cleanup cancelled."
        fi
        ;;

    help|--help|-h)
        usage
        ;;

    *)
        print_error "Unknown command: $COMMAND"
        echo
        usage
        exit 1
        ;;
esac
