from loguru import logger
from ddm_project import hello

def main():
    logger.info(hello())

if __name__ == "__main__":
    main()
