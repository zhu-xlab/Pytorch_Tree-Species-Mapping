import os
import shutil


def mkdir_if_not_exist(dir_name, is_delete=False):
    """
    :param dir_name:
    :param is_delete:
    :return: True or false
    """
    try:
        if is_delete:
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)
                print('[Info] Dir "%s" exists, delete.' % dir_name)

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print('[Info] Dir "%s" does not exist, create.' % dir_name)
        return True
    except Exception as e:
        print('[Exception] %s' % e)
        return False
