from scripts_py.agent_module.openai_llm import OpenaiConfig, OpenaiLlm

class BookReader(PdfReader):
    BookTemplates = {
        'extract_page_header_regex': [{
        "role": "system",
        "content": "You are an expert editor and text formatting assistant.",
    },
    {
        "role": "user",
        "content": """
### Object
Given some page header examples, write regex in order to filter out page header from document

### Examples
{examples}

### Additional Notes
1. Assume each example is always in single line
2. Assume there are non-certain space between the words of example

### Output Format
1. Your answer must be Json, start with {{ and end with }}

### Output
Your Answer
"""}]
    }
    
    def __init__(
        self, 
        file, openai_config: OpenaiConfig,
        mode = 'page', extract_images=False,
        contents_pages:list[int] = [],
        page_header_examples:list[str] = [],
        page_footing_examples:list[str] = [],
        start_page: int = 0,
        finish_page: int = -1
    ):
        super().__init__(file, mode, extract_images)
        self.llm = OpenaiLlm(openai_config)
        
        self.page_header_examples = page_header_examples
        if len(page_header_examples) > 0:
            res = self.extract_page_header_regex()
            if res['is_success']:
                print(f'[Success extract page header regex]: {res["content"]}')
            else:
                print(f'[Error extract page header regex]: {res["error"]}')
                return
        
        if len(contents_pages) > 0:
            self.extract_book_contents()
    
    def extract_page_header_regex(self) -> dict:
        template_chat: list[str] = self.BookTemplates['extract_page_header_regex']
        template_chat[1]['content'] = template_chat[1]['content'].format(examples='\n'.join(self.page_header_examples))
        response_message, metadata = self.llm.chat_completion(template_chat)
        try:
            if metadata['finish_reason'] == 'stop':
                return {'is_success': True, 'content': json.loads(response_message)['regex']}
            return {'is_success': False, 'error': metadata['finish_reason']}
        except Exception as e:
            return {'is_success': False, 'error': str(e)}
    
    def extract_book_contents(self):
        pass


def test_book():
    book_file = '/home/admin123456/Downloads/OpenFOAMUserGuide-A4.pdf'
    
    openai_config = OpenaiConfig(
        api_key='123456',
        model_name='Qwen3_Chat',
        base_url='http://192.168.110.79:8889/v1'
    )
    
    reader = BookReader(
        file=book_file, openai_config=openai_config,
        page_header_examples=[
            'U-18 Tutorials',
            'U-20 Tutorials',
            '2.1 Backward-facing step U-21',
            '5.6 Mesh Zones U165'
            'U-166 Mesh generation and conversion'
        ]
    )


if __name__ == "__main__":
    test_book()
